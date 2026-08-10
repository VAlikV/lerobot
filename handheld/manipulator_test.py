"""Replay one Quest delta-action episode on a KUKA iiwa."""

from pathlib import Path
import threading
import time

import matplotlib.pyplot as plt
import multiprocess.resource_tracker
import numpy as np
from scipy.spatial.transform import Rotation

# multiprocess 0.70.18 is incompatible with the RLock implementation in
# CPython 3.12.0. LeRobot imports multiprocess through Hugging Face datasets.
if not hasattr(threading.RLock(), "_recursion_count"):
    def _skip_incompatible_resource_tracker_finalizer(_resource_tracker: object) -> None:
        pass

    multiprocess.resource_tracker.ResourceTracker.__del__ = _skip_incompatible_resource_tracker_finalizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.utils.robot_utils import precise_sleep


DATASET_REPO_ID = "local/quest_test_dataset"
EPISODE_INDEX = 0

URDF_PATH = Path(__file__).parents[1] / "src" / "lerobot" / "robots" / "kuka_iiwa" / "iiwa2_gripper.urdf"
GRIPPER_PORT = "/dev/ttyACM0"
PLOT_PATH = Path(__file__).parent / f"replay_trajectory_episode_{EPISODE_INDEX:03d}.png"
ORIENTATION_PLOT_PATH = Path(__file__).parent / f"replay_orientation_episode_{EPISODE_INDEX:03d}.png"

# Leave these at 1.0 to reproduce the recorded trajectory exactly.
POSITION_SCALE = 1.0
ROTATION_SCALE = 1.0

EXPECTED_ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
]


def plot_trajectories(dataset_xyz: np.ndarray, actual_xyz: np.ndarray, fps: int) -> None:
    """Plot the desired dataset trajectory against the measured KUKA trajectory."""
    frame_count = min(len(dataset_xyz), len(actual_xyz))
    dataset_xyz = dataset_xyz[:frame_count]
    actual_xyz = actual_xyz[:frame_count]
    time_s = np.arange(frame_count) / fps

    figure = plt.figure(figsize=(13, 9))
    trajectory_axis = figure.add_subplot(2, 2, 1, projection="3d")
    trajectory_axis.plot(*dataset_xyz.T, label="Dataset", linewidth=2)
    trajectory_axis.plot(*actual_xyz.T, label="KUKA measured", linewidth=2)
    trajectory_axis.set_xlabel("x, m")
    trajectory_axis.set_ylabel("y, m")
    trajectory_axis.set_zlabel("z, m")
    trajectory_axis.set_title("Cartesian trajectory")
    trajectory_axis.legend()

    for plot_index, (axis_name, coordinate_index) in enumerate(zip("xyz", range(3)), start=2):
        axis = figure.add_subplot(2, 2, plot_index)
        axis.plot(time_s, dataset_xyz[:, coordinate_index], label="Dataset")
        axis.plot(time_s, actual_xyz[:, coordinate_index], label="KUKA measured")
        axis.set_xlabel("Time, s")
        axis.set_ylabel(f"{axis_name}, m")
        axis.grid(True)
        axis.legend()

    figure.suptitle(f"Replay trajectory, episode {EPISODE_INDEX}")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    print(f"Trajectory plot saved to {PLOT_PATH}")


def plot_orientations(dataset_rpy: np.ndarray, actual_rpy: np.ndarray, fps: int) -> None:
    """Plot relative roll, pitch and yaw from the dataset and KUKA."""
    frame_count = min(len(dataset_rpy), len(actual_rpy))
    dataset_rpy = dataset_rpy[:frame_count]
    actual_rpy = actual_rpy[:frame_count]
    time_s = np.arange(frame_count) / fps

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for axis, axis_name, coordinate_index in zip(axes, ("roll", "pitch", "yaw"), range(3)):
        axis.plot(time_s, dataset_rpy[:, coordinate_index], label="Dataset")
        axis.plot(time_s, actual_rpy[:, coordinate_index], label="KUKA measured")
        axis.set_ylabel(f"{axis_name}, rad")
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time, s")
    figure.suptitle(f"Replay orientation, episode {EPISODE_INDEX}")
    figure.tight_layout()
    figure.savefig(ORIENTATION_PLOT_PATH, dpi=150)
    print(f"Orientation plot saved to {ORIENTATION_PLOT_PATH}")


def main() -> None:
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        episodes=[EPISODE_INDEX],
        download_videos=False,
    )

    action_names = dataset.features["action"].get("names")
    if action_names != EXPECTED_ACTION_NAMES:
        raise ValueError(
            "This replay script expects Quest delta actions "
            f"{EXPECTED_ACTION_NAMES}, but the dataset contains {action_names}."
        )

    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float64)
    if len(actions) == 0:
        raise ValueError(f"Episode {EPISODE_INDEX} contains no actions")

    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float64)
    dataset_xyz = (states[:, :3] - states[0, :3]) * POSITION_SCALE
    dataset_rotation = Rotation.identity()
    dataset_rpy = []
    for action in actions:
        dataset_rotation = dataset_rotation * Rotation.from_euler(
            "xyz",
            action[3:6] * ROTATION_SCALE,
            degrees=False,
        )
        dataset_rpy.append(dataset_rotation.as_euler("xyz", degrees=False))
    dataset_rpy = np.asarray(dataset_rpy)

    actual_xyz: list[np.ndarray] = []
    actual_rpy: list[np.ndarray] = []

    robot = KukaIiwa(
        KukaIiwaConfig(
            urdf_path=str(URDF_PATH),
            gripper_port=GRIPPER_PORT,
            cameras={},
        )
    )

    print(
        f"Dataset: {DATASET_REPO_ID}\n"
        f"Episode: {EPISODE_INDEX}\n"
        f"Frames: {len(actions)}, FPS: {dataset.fps}"
    )
    input("Press Enter to connect to KUKA and start replay (Ctrl+C to cancel) ...")

    try:
        print("Connecting to KUKA iiwa ...")
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

        period_s = 1.0 / dataset.fps
        for frame_index, action in enumerate(actions):
            cycle_start = time.perf_counter()
            action = np.asarray(action, dtype=np.float64)

            target_position += action[:3] * POSITION_SCALE
            rotation_delta = Rotation.from_euler(
                "xyz",
                action[3:6] * ROTATION_SCALE,
                degrees=False,
            )
            target_rotation = target_rotation * rotation_delta
            roll, pitch, yaw = target_rotation.as_euler("xyz", degrees=False)

            # Quest: 0=open, 1=closed. KUKA: +1=open, -1=closed.
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

            print(f"\rFrame {frame_index + 1}/{len(actions)}", end="", flush=True)
            precise_sleep(max(period_s - (time.perf_counter() - cycle_start), 0.0))

            measured = robot.get_observation()
            actual_xyz.append(
                np.array(
                    [measured["x.pos"], measured["y.pos"], measured["z.pos"]],
                    dtype=np.float64,
                )
                - initial_position
            )
            measured_rotation = Rotation.from_euler(
                "xyz",
                [measured["roll.pos"], measured["pitch.pos"], measured["yaw.pos"]],
                degrees=False,
            )
            relative_rotation = initial_rotation.inv() * measured_rotation
            actual_rpy.append(relative_rotation.as_euler("xyz", degrees=False))

        print("\nReplay finished.")
    except KeyboardInterrupt:
        print("\nReplay stopped.")
    finally:
        if robot.is_connected:
            robot.disconnect()

    if actual_xyz:
        plot_trajectories(dataset_xyz, np.stack(actual_xyz), dataset.fps)
        plot_orientations(dataset_rpy, np.stack(actual_rpy), dataset.fps)
        plt.show()


if __name__ == "__main__":
    main()
