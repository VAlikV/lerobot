"""Run ACT and compare its trajectory with the average training trajectory."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import manipulator_act_test as act_test
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.utils.robot_utils import precise_sleep


OUTLIER_SIGMA = 2.0
MIN_POSITION_BAND_M = 0.005
MIN_ROTATION_BAND_RAD = 0.02
LIVE_PLOT_EVERY_N_STEPS = 5
PLOT_PATH = Path(__file__).parent / "act_average_trajectory_comparison.png"


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def load_average_trajectory() -> tuple[np.ndarray, np.ndarray, int]:
    """Return circular mean and standard deviation by episode frame index."""
    dataset = LeRobotDataset(act_test.DATASET_REPO_ID, download_videos=False)
    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float64)
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    episodes = [states[episode_indices == index, :6] for index in np.unique(episode_indices)]

    max_frames = max(len(episode) for episode in episodes)
    stacked = np.full((len(episodes), max_frames, 6), np.nan, dtype=np.float64)
    for episode_index, episode in enumerate(episodes):
        stacked[episode_index, : len(episode)] = episode

    mean = np.empty((max_frames, 6), dtype=np.float64)
    std = np.empty((max_frames, 6), dtype=np.float64)
    mean[:, :3] = np.nanmean(stacked[:, :, :3], axis=0)
    std[:, :3] = np.nanstd(stacked[:, :, :3], axis=0)

    angles = stacked[:, :, 3:6]
    mean_angles = np.arctan2(np.nanmean(np.sin(angles), axis=0), np.nanmean(np.cos(angles), axis=0))
    angular_errors = wrap_angles(angles - mean_angles[None, :, :])
    mean[:, 3:6] = mean_angles
    std[:, 3:6] = np.nanstd(angular_errors, axis=0)
    return mean, std, len(episodes)


class AverageTrajectoryPlot:
    def __init__(self, mean: np.ndarray, std: np.ndarray, episode_count: int) -> None:
        plt.ion()
        self.mean = mean
        self.std = std
        self.time_s = np.arange(len(mean)) / act_test.CONTROL_FPS
        self.figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
        self.predicted_lines = []
        self.measured_lines = []

        for index, axis_name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            axis = axes[index // 3, index % 3]
            band = np.maximum(
                OUTLIER_SIGMA * std[:, index],
                MIN_POSITION_BAND_M if index < 3 else MIN_ROTATION_BAND_RAD,
            )
            axis.fill_between(
                self.time_s,
                mean[:, index] - band,
                mean[:, index] + band,
                color="tab:blue",
                alpha=0.18,
                label=f"Mean ± {OUTLIER_SIGMA:g}σ",
            )
            axis.plot(self.time_s, mean[:, index], color="tab:blue", label="Dataset mean")
            predicted_line, = axis.plot([], [], color="tab:red", label="ACT target")
            measured_line, = axis.plot([], [], color="tab:orange", label="KUKA measured")
            axis.set_title(axis_name)
            axis.set_ylabel("m" if index < 3 else "rad")
            if index >= 3:
                axis.set_xlabel("Time, s")
            axis.grid(True)
            axis.legend()
            self.predicted_lines.append(predicted_line)
            self.measured_lines.append(measured_line)

        self.figure.suptitle(f"ACT vs average of {episode_count} training episodes")
        self.figure.tight_layout()
        self.figure.show()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update(self, predicted: list[np.ndarray], measured: list[np.ndarray]) -> None:
        if not predicted or not measured:
            return
        frame_count = min(len(predicted), len(measured))
        time_s = np.arange(frame_count) / act_test.CONTROL_FPS
        predicted_array = np.asarray(predicted[:frame_count])
        measured_array = np.asarray(measured[:frame_count])
        for index, (predicted_line, measured_line) in enumerate(
            zip(self.predicted_lines, self.measured_lines)
        ):
            predicted_line.set_data(time_s, predicted_array[:, index])
            measured_line.set_data(time_s, measured_array[:, index])
            predicted_line.axes.relim()
            predicted_line.axes.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def save(self) -> None:
        self.figure.savefig(PLOT_PATH, dpi=150)


class LiveCameraPlot:
    """Show the RGB image currently passed to the ACT policy."""

    def __init__(self, image: np.ndarray) -> None:
        self.figure, self.axis = plt.subplots(figsize=(8, 6))
        self.image_artist = self.axis.imshow(image)
        self.axis.set_title("Live ACT input: observation.images.handheld")
        self.axis.axis("off")
        self.figure.tight_layout()
        self.figure.show()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update(self, image: np.ndarray, step: int) -> None:
        self.image_artist.set_data(image)
        self.axis.set_title(f"Live ACT input — step {step}")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


def trajectory_outlier_axes(step: int, pose: np.ndarray, mean: np.ndarray, std: np.ndarray) -> list[str]:
    if step >= len(mean):
        return []
    error = pose - mean[step]
    error[3:6] = wrap_angles(error[3:6])
    threshold = np.maximum(
        OUTLIER_SIGMA * std[step],
        np.array([MIN_POSITION_BAND_M] * 3 + [MIN_ROTATION_BAND_RAD] * 3),
    )
    names = ("x", "y", "z", "roll", "pitch", "yaw")
    return [names[index] for index in np.flatnonzero(np.abs(error) > threshold)]


def main() -> None:
    mean, std, episode_count = load_average_trajectory()
    metadata = LeRobotDatasetMetadata(act_test.DATASET_REPO_ID)
    device = torch.device(
        act_test.DEVICE
        if torch.cuda.is_available() or act_test.DEVICE == "cpu"
        else "cpu"
    )

    policy = ACTPolicy.from_pretrained(act_test.MODEL_PATH)
    policy.config.device = str(device)
    policy.config.n_action_steps = act_test.N_ACTION_STEPS
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )

    robot = KukaIiwa(
        KukaIiwaConfig(
            urdf_path=str(act_test.URDF_PATH),
            gripper_port=act_test.GRIPPER_PORT,
            cameras={
                "handheld": OpenCVCameraConfig(
                    index_or_path=act_test.CAMERA_INDEX,
                    width=act_test.CAMERA_WIDTH,
                    height=act_test.CAMERA_HEIGHT,
                    fps=act_test.CAMERA_FPS,
                    warmup_s=5,
                )
            },
            resolution=(act_test.IMAGE_WIDTH, act_test.IMAGE_HEIGHT),
        )
    )

    print(f"Averaged {episode_count} episodes. Model: {act_test.MODEL_PATH}")
    input("Press Enter to connect to KUKA and run ACT (Ctrl+C to cancel) ...")

    predicted_poses: list[np.ndarray] = []
    measured_poses: list[np.ndarray] = []
    plot = AverageTrajectoryPlot(mean, std, episode_count)
    camera_plot = None

    try:
        robot.connect()
        observation = robot.get_observation()
        camera_plot = LiveCameraPlot(np.asarray(observation["handheld"]))
        initial_position = np.array(
            [observation["x.pos"], observation["y.pos"], observation["z.pos"]],
            dtype=np.float64,
        )
        initial_rotation = Rotation.from_euler(
            "xyz",
            [observation["roll.pos"], observation["pitch.pos"], observation["yaw.pos"]],
            degrees=False,
        )
        target_position = initial_position.copy()
        target_rotation = initial_rotation
        period_s = 1.0 / act_test.CONTROL_FPS

        for step_index in range(act_test.MAX_STEPS):
            cycle_start = time.perf_counter()
            if act_test.REANCHOR_TARGET_EACH_CHUNK and step_index % act_test.N_ACTION_STEPS == 0:
                target_position = np.array(
                    [observation["x.pos"], observation["y.pos"], observation["z.pos"]],
                    dtype=np.float64,
                )
                target_rotation = Rotation.from_euler(
                    "xyz",
                    [observation["roll.pos"], observation["pitch.pos"], observation["yaw.pos"]],
                    degrees=False,
                )

            policy_observation = act_test.make_policy_observation(
                observation, initial_position, initial_rotation
            )
            with torch.no_grad():
                action = policy.select_action(preprocessor(policy_observation))
            action = postprocessor(action).squeeze(0).detach().cpu().numpy().astype(np.float64)

            target_position += np.clip(
                action[:3] * act_test.POSITION_ACTION_SCALE,
                -act_test.MAX_POSITION_DELTA_M,
                act_test.MAX_POSITION_DELTA_M,
            )
            target_rotation = target_rotation * Rotation.from_euler(
                "xyz",
                np.clip(
                    action[3:6] * act_test.ROTATION_ACTION_SCALE,
                    -act_test.MAX_ROTATION_DELTA_RAD,
                    act_test.MAX_ROTATION_DELTA_RAD,
                ),
            )
            target_relative_rotation = initial_rotation.inv() * target_rotation
            predicted_pose = np.concatenate(
                [
                    target_position - initial_position,
                    target_relative_rotation.as_euler("xyz", degrees=False),
                ]
            )
            predicted_poses.append(predicted_pose)

            target_rpy = target_rotation.as_euler("xyz", degrees=False)
            robot.send_action(
                {
                    "x.pos": float(target_position[0]),
                    "y.pos": float(target_position[1]),
                    "z.pos": float(target_position[2]),
                    "roll.pos": float(target_rpy[0]),
                    "pitch.pos": float(target_rpy[1]),
                    "yaw.pos": float(target_rpy[2]),
                    "gripper.pos": -1.0 if action[6] >= 0.5 else 1.0,
                }
            )

            observation = robot.get_observation()
            measured_position = np.array(
                [observation["x.pos"], observation["y.pos"], observation["z.pos"]]
            )
            measured_rotation = Rotation.from_euler(
                "xyz",
                [observation["roll.pos"], observation["pitch.pos"], observation["yaw.pos"]],
            )
            measured_poses.append(
                np.concatenate(
                    [
                        measured_position - initial_position,
                        (initial_rotation.inv() * measured_rotation).as_euler("xyz"),
                    ]
                )
            )

            outlier_axes = trajectory_outlier_axes(step_index, predicted_pose, mean, std)
            if outlier_axes:
                print(f"\nStep {step_index}: ACT trajectory outside band: {', '.join(outlier_axes)}")

            if step_index % LIVE_PLOT_EVERY_N_STEPS == 0:
                plot.update(predicted_poses, measured_poses)
                camera_plot.update(np.asarray(observation["handheld"]), step_index)
            print(f"\rACT step {step_index + 1}/{act_test.MAX_STEPS}", end="", flush=True)
            precise_sleep(max(period_s - (time.perf_counter() - cycle_start), 0.0))
    except KeyboardInterrupt:
        print("\nACT run stopped.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        plot.update(predicted_poses, measured_poses)
        plot.save()
        print(f"Comparison plot saved to {PLOT_PATH}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
