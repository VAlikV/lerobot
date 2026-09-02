"""Run a handheld-camera ACT policy on a KUKA iiwa."""

from pathlib import Path
import threading
import time

import matplotlib.pyplot as plt
import multiprocess.resource_tracker
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Compatibility workaround for multiprocess 0.70.18 on CPython 3.12.0.
if not hasattr(threading.RLock(), "_recursion_count"):
    def _skip_incompatible_resource_tracker_finalizer(_resource_tracker: object) -> None:
        pass

    multiprocess.resource_tracker.ResourceTracker.__del__ = _skip_incompatible_resource_tracker_finalizer

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.utils.robot_utils import precise_sleep


MODEL_PATH = "outputs/quest_tasks/act_red_cube_pick_and_place_2_0.25/70000"
DATASET_REPO_ID = "local/quest_red_cube_pick_and_place_2_0.25"
DEVICE = "cuda"

CAMERA_FPS = 30
CONTROL_FPS = 30
MAX_STEPS = CONTROL_FPS * 60
N_ACTION_STEPS = 5
REANCHOR_TARGET_EACH_CHUNK = False

# Scale policy pose deltas without changing camera/model FPS. Values below 1
# slow Cartesian motion; the gripper command is not scaled.
POSITION_ACTION_SCALE = 1
ROTATION_ACTION_SCALE = 1

URDF_PATH = Path(__file__).parents[1] / "src" / "lerobot" / "robots" / "kuka_iiwa" / "iiwa2_gripper.urdf"
GRIPPER_PORT = "/dev/ttyACM0"
CAMERA_INDEX = 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

LIVE_PLOT = False
LIVE_PLOT_EVERY_N_STEPS = 5
RELATIVE_POSE_PLOT_PATH = Path(__file__).parent / "act_relative_pose.png"
ABSOLUTE_POSE_PLOT_PATH = Path(__file__).parent / "act_absolute_pose.png"

# Per-control-step safety clipping for unexpected policy outputs.
MAX_POSITION_DELTA_M = 0.1
MAX_ROTATION_DELTA_RAD = 0.1

EXPECTED_ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
]


class LivePolicyPlot:
    """Display the ACT target and measured KUKA pose during execution."""

    def __init__(self, title: str) -> None:
        plt.ion()
        self.figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
        self.target_lines = []
        self.measured_lines = []

        for index, axis_name in enumerate("xyz"):
            axis = axes[0, index]
            target_line, = axis.plot([], [], label="ACT target", color="tab:blue")
            measured_line, = axis.plot([], [], label="KUKA measured", color="tab:orange")
            axis.set_title(f"Position {axis_name}")
            axis.set_ylabel("m")
            axis.grid(True)
            axis.legend()
            self.target_lines.append(target_line)
            self.measured_lines.append(measured_line)

        for index, axis_name in enumerate(("roll", "pitch", "yaw"), start=3):
            axis = axes[1, index - 3]
            target_line, = axis.plot([], [], label="ACT target", color="tab:blue")
            measured_line, = axis.plot([], [], label="KUKA measured", color="tab:orange")
            axis.set_title(axis_name)
            axis.set_xlabel("Time, s")
            axis.set_ylabel("rad")
            axis.grid(True)
            axis.legend()
            self.target_lines.append(target_line)
            self.measured_lines.append(measured_line)

        self.figure.suptitle(title)
        self.figure.tight_layout()
        self.figure.show()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update(self, target_pose: list[np.ndarray], measured_pose: list[np.ndarray]) -> None:
        if not target_pose or not measured_pose:
            return

        frame_count = min(len(target_pose), len(measured_pose))
        time_s = np.arange(frame_count) / CONTROL_FPS
        target = np.asarray(target_pose[:frame_count])
        measured = np.asarray(measured_pose[:frame_count])

        for index, (target_line, measured_line) in enumerate(
            zip(self.target_lines, self.measured_lines)
        ):
            target_line.set_data(time_s, target[:, index])
            measured_line.set_data(time_s, measured[:, index])
            target_line.axes.relim()
            target_line.axes.autoscale_view()

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def save(self, path: Path) -> None:
        self.figure.savefig(path, dpi=150)


def make_policy_observation(
    robot_observation: dict,
    initial_position: np.ndarray,
    initial_rotation: Rotation,
) -> dict[str, torch.Tensor]:
    current_position = np.array(
        [
            robot_observation["x.pos"],
            robot_observation["y.pos"],
            robot_observation["z.pos"],
        ],
        dtype=np.float64,
    )
    current_rotation = Rotation.from_euler(
        "xyz",
        [
            robot_observation["roll.pos"],
            robot_observation["pitch.pos"],
            robot_observation["yaw.pos"],
        ],
        degrees=False,
    )
    relative_rotation = initial_rotation.inv() * current_rotation
    state = np.concatenate(
        [
            current_position - initial_position,
            relative_rotation.as_euler("xyz", degrees=False),
        ]
    ).astype(np.float32)

    image = np.asarray(robot_observation["handheld"])
    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.handheld": image_tensor,
    }


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    if metadata.features["action"].get("names") != EXPECTED_ACTION_NAMES:
        raise ValueError(
            f"Expected action names {EXPECTED_ACTION_NAMES}, "
            f"got {metadata.features['action'].get('names')}."
        )

    policy = ACTPolicy.from_pretrained(MODEL_PATH)
    if N_ACTION_STEPS > policy.config.chunk_size:
        raise ValueError(
            f"N_ACTION_STEPS={N_ACTION_STEPS} exceeds ACT chunk_size={policy.config.chunk_size}."
        )
    policy.config.device = str(device)
    policy.config.n_action_steps = N_ACTION_STEPS
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )

    robot = KukaIiwa(
        KukaIiwaConfig(
            urdf_path=str(URDF_PATH),
            gripper_port=GRIPPER_PORT,
            cameras={
                "handheld": OpenCVCameraConfig(
                    index_or_path=CAMERA_INDEX,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=CAMERA_FPS,
                    warmup_s=5,
                )
            },
            resolution=(IMAGE_WIDTH, IMAGE_HEIGHT),
        )
    )

    print(
        f"Model: {MODEL_PATH}\n"
        f"Dataset stats: {DATASET_REPO_ID}\n"
        f"Device: {device}, action steps: {N_ACTION_STEPS}"
    )
    input("Press Enter to connect to KUKA and run ACT (Ctrl+C to cancel) ...")

    target_relative_pose: list[np.ndarray] = []
    measured_relative_pose: list[np.ndarray] = []
    target_absolute_pose: list[np.ndarray] = []
    measured_absolute_pose: list[np.ndarray] = []
    relative_plot = None
    absolute_plot = None

    try:
        robot.connect()
        observation = robot.get_observation()
        target_position = np.array(
            [observation["x.pos"], observation["y.pos"], observation["z.pos"]],
            dtype=np.float64,
        )
        initial_position = target_position.copy()
        target_rotation = Rotation.from_euler(
            "xyz",
            [observation["roll.pos"], observation["pitch.pos"], observation["yaw.pos"]],
            degrees=False,
        )
        initial_rotation = target_rotation
        if LIVE_PLOT:
            relative_plot = LivePolicyPlot("ACT live relative pose")
            absolute_plot = LivePolicyPlot("ACT live absolute pose")

        period_s = 1.0 / CONTROL_FPS
        for step_index in range(MAX_STEPS):
            cycle_start = time.perf_counter()

            # Every new chunk is predicted from this measured observation.
            # Use the same pose as its delta-integration baseline instead of
            # carrying tracking error over from the preceding open-loop chunk.
            if REANCHOR_TARGET_EACH_CHUNK and step_index % N_ACTION_STEPS == 0:
                target_position = np.array(
                    [observation["x.pos"], observation["y.pos"], observation["z.pos"]],
                    dtype=np.float64,
                )
                target_rotation = Rotation.from_euler(
                    "xyz",
                    [observation["roll.pos"], observation["pitch.pos"], observation["yaw.pos"]],
                    degrees=False,
                )

            policy_observation = make_policy_observation(
                observation,
                initial_position,
                initial_rotation,
            )
            policy_observation = preprocessor(policy_observation)

            with torch.no_grad():
                action = policy.select_action(policy_observation)
            action = postprocessor(action).squeeze(0).detach().cpu().numpy().astype(np.float64)
            if action.shape != (7,):
                raise ValueError(f"Expected ACT action shape (7,), got {action.shape}.")

            position_delta = np.clip(
                action[:3] * POSITION_ACTION_SCALE,
                -MAX_POSITION_DELTA_M,
                MAX_POSITION_DELTA_M,
            )
            rotation_delta = np.clip(
                action[3:6] * ROTATION_ACTION_SCALE,
                -MAX_ROTATION_DELTA_RAD,
                MAX_ROTATION_DELTA_RAD,
            )
            target_position += position_delta
            target_rotation = target_rotation * Rotation.from_euler(
                "xyz",
                rotation_delta,
                degrees=False,
            )
            roll, pitch, yaw = target_rotation.as_euler("xyz", degrees=False)
            target_relative_rotation = initial_rotation.inv() * target_rotation
            target_relative_pose.append(
                np.concatenate(
                    [
                        target_position - initial_position,
                        target_relative_rotation.as_euler("xyz", degrees=False),
                    ]
                )
            )
            target_absolute_pose.append(
                np.array(
                    [target_position[0], target_position[1], target_position[2], roll, pitch, yaw],
                    dtype=np.float64,
                )
            )
            gripper_position = -1.0 if action[6] >= 0.5 else 1.0

            robot.send_action(
                {
                    "x.pos": float(target_position[0]),
                    "y.pos": float(target_position[1]),
                    "z.pos": float(target_position[2]),
                    "roll.pos": float(roll),
                    "pitch.pos": float(pitch),
                    "yaw.pos": float(yaw),
                    "gripper.pos": gripper_position,
                }
            )

            observation = robot.get_observation()
            measured_position = np.array(
                [observation["x.pos"], observation["y.pos"], observation["z.pos"]],
                dtype=np.float64,
            )
            measured_rpy = np.array(
                [observation["roll.pos"], observation["pitch.pos"], observation["yaw.pos"]],
                dtype=np.float64,
            )
            measured_rotation = Rotation.from_euler("xyz", measured_rpy, degrees=False)
            measured_relative_rotation = initial_rotation.inv() * measured_rotation
            measured_relative_pose.append(
                np.concatenate(
                    [
                        measured_position - initial_position,
                        measured_relative_rotation.as_euler("xyz", degrees=False),
                    ]
                )
            )
            measured_absolute_pose.append(np.concatenate([measured_position, measured_rpy]))

            if relative_plot is not None and (
                step_index % LIVE_PLOT_EVERY_N_STEPS == 0
                or step_index == MAX_STEPS - 1
            ):
                relative_plot.update(target_relative_pose, measured_relative_pose)
                absolute_plot.update(target_absolute_pose, measured_absolute_pose)

            print(f"\rACT step {step_index + 1}/{MAX_STEPS}", end="", flush=True)
            precise_sleep(max(period_s - (time.perf_counter() - cycle_start), 0.0))

        print("\nACT run finished.")
    except KeyboardInterrupt:
        print("\nACT run stopped.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        if relative_plot is not None:
            relative_plot.update(target_relative_pose, measured_relative_pose)
            relative_plot.save(RELATIVE_POSE_PLOT_PATH)
            print(f"Relative pose plot saved to {RELATIVE_POSE_PLOT_PATH}")
        if absolute_plot is not None:
            absolute_plot.update(target_absolute_pose, measured_absolute_pose)
            absolute_plot.save(ABSOLUTE_POSE_PLOT_PATH)
            print(f"Absolute pose plot saved to {ABSOLUTE_POSE_PLOT_PATH}")

    if relative_plot is not None:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
