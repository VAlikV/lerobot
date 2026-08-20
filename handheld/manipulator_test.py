"""Replay one Quest delta-action episode on a KUKA iiwa."""

import csv
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


DATASET_REPO_ID = "local/quest_red_cube_pick_and_place_2_0.25"
EPISODE_INDEX = 2

URDF_PATH = Path(__file__).parents[1] / "src" / "lerobot" / "robots" / "kuka_iiwa" / "iiwa2_gripper.urdf"
GRIPPER_PORT = "/dev/ttyACM0"
PLOT_PATH = Path(__file__).parent / f"replay_trajectory_episode_{EPISODE_INDEX:03d}.png"
ORIENTATION_PLOT_PATH = Path(__file__).parent / f"replay_orientation_episode_{EPISODE_INDEX:03d}.png"
ABSOLUTE_POSE_PLOT_PATH = Path(__file__).parent / f"replay_absolute_pose_episode_{EPISODE_INDEX:03d}.png"
POSE_LOG_PATH = Path(__file__).parent / f"replay_pose_episode_{EPISODE_INDEX:03d}.csv"

# Leave these at 1.0 to reproduce the recorded trajectory exactly.
POSITION_SCALE = 1.0
ROTATION_SCALE = 1.0

# Updating matplotlib on every control cycle can noticeably slow replay down.
LIVE_PLOT = False
LIVE_PLOT_EVERY_N_FRAMES = 5
CONSOLE_LOG_EVERY_N_FRAMES = 1

EXPECTED_ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
]


class LiveReplayPlot:
    """Show desired and measured Cartesian pose while replay is running."""

    def __init__(
        self,
        dataset_xyz: np.ndarray,
        dataset_rpy: np.ndarray,
        fps: int,
        *,
        title: str = "Relative pose",
        target_label: str = "Dataset",
    ) -> None:
        plt.ion()
        self.dataset_xyz = dataset_xyz
        self.dataset_rpy = dataset_rpy
        self.time_s = np.arange(len(dataset_xyz)) / fps

        self.figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
        self.position_lines = []
        self.orientation_lines = []

        for index, axis_name in enumerate("xyz"):
            axis = axes[0, index]
            axis.plot(
                self.time_s,
                dataset_xyz[:, index],
                label=target_label,
                color="tab:blue",
            )
            measured_line, = axis.plot([], [], label="KUKA measured", color="tab:orange")
            cursor = axis.axvline(0.0, color="black", alpha=0.25)
            axis.set_title(f"Position {axis_name}")
            axis.set_ylabel("m")
            axis.grid(True)
            axis.legend()
            self.position_lines.append((measured_line, cursor))

        for index, axis_name in enumerate(("roll", "pitch", "yaw")):
            axis = axes[1, index]
            axis.plot(
                self.time_s,
                dataset_rpy[:, index],
                label=target_label,
                color="tab:blue",
            )
            measured_line, = axis.plot([], [], label="KUKA measured", color="tab:orange")
            cursor = axis.axvline(0.0, color="black", alpha=0.25)
            axis.set_title(axis_name)
            axis.set_xlabel("Time, s")
            axis.set_ylabel("rad")
            axis.grid(True)
            axis.legend()
            self.orientation_lines.append((measured_line, cursor))

        self.figure.suptitle(f"{title}, episode {EPISODE_INDEX}")
        self.figure.tight_layout()
        self.figure.show()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update(
        self,
        actual_xyz: list[np.ndarray],
        actual_rpy: list[np.ndarray],
    ) -> None:
        if not actual_xyz:
            return

        measured_xyz = np.asarray(actual_xyz)
        measured_rpy = np.asarray(actual_rpy)
        measured_time_s = self.time_s[: len(measured_xyz)]
        current_time_s = measured_time_s[-1]

        for index, (line, cursor) in enumerate(self.position_lines):
            line.set_data(measured_time_s, measured_xyz[:, index])
            cursor.set_xdata([current_time_s, current_time_s])
            line.axes.relim()
            line.axes.autoscale_view(scalex=False, scaley=True)

        for index, (line, cursor) in enumerate(self.orientation_lines):
            line.set_data(measured_time_s, measured_rpy[:, index])
            cursor.set_xdata([current_time_s, current_time_s])
            line.axes.relim()
            line.axes.autoscale_view(scalex=False, scaley=True)

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self) -> None:
        plt.ioff()


def build_absolute_targets(
    actions: np.ndarray,
    initial_position: np.ndarray,
    initial_rotation: Rotation,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate dataset deltas into the exact absolute poses sent to IK."""
    position = initial_position.copy()
    rotation = initial_rotation
    target_xyz = []
    target_rpy = []

    for action in actions:
        position = position + action[:3] * POSITION_SCALE
        rotation = rotation * Rotation.from_euler(
            "xyz",
            action[3:6] * ROTATION_SCALE,
            degrees=False,
        )
        target_xyz.append(position.copy())
        target_rpy.append(rotation.as_euler("xyz", degrees=False))

    return np.asarray(target_xyz), np.asarray(target_rpy)


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


def plot_absolute_pose(
    target_xyz: np.ndarray,
    measured_xyz: np.ndarray,
    target_rpy: np.ndarray,
    measured_rpy: np.ndarray,
    fps: int,
) -> None:
    """Save the absolute target sent to IK and the measured KUKA pose."""
    frame_count = min(len(target_xyz), len(measured_xyz))
    target_xyz = target_xyz[:frame_count]
    measured_xyz = measured_xyz[:frame_count]
    target_rpy = target_rpy[:frame_count]
    measured_rpy = measured_rpy[:frame_count]
    time_s = np.arange(frame_count) / fps

    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
    for index, axis_name in enumerate("xyz"):
        axis = axes[0, index]
        axis.plot(time_s, target_xyz[:, index], label="IK target")
        axis.plot(time_s, measured_xyz[:, index], label="KUKA measured")
        axis.set_title(f"Absolute position {axis_name}")
        axis.set_ylabel("m")
        axis.grid(True)
        axis.legend()

    for index, axis_name in enumerate(("roll", "pitch", "yaw")):
        axis = axes[1, index]
        axis.plot(time_s, target_rpy[:, index], label="IK target")
        axis.plot(time_s, measured_rpy[:, index], label="KUKA measured")
        axis.set_title(f"Absolute {axis_name}")
        axis.set_xlabel("Time, s")
        axis.set_ylabel("rad")
        axis.grid(True)
        axis.legend()

    figure.suptitle(f"Absolute pose, episode {EPISODE_INDEX}")
    figure.tight_layout()
    figure.savefig(ABSOLUTE_POSE_PLOT_PATH, dpi=150)
    print(f"Absolute pose plot saved to {ABSOLUTE_POSE_PLOT_PATH}")


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
    actual_absolute_xyz: list[np.ndarray] = []
    actual_absolute_rpy: list[np.ndarray] = []
    absolute_target_xyz = np.empty((0, 3), dtype=np.float64)
    absolute_target_rpy = np.empty((0, 3), dtype=np.float64)
    live_plot = (
        LiveReplayPlot(dataset_xyz, dataset_rpy, dataset.fps, title="Live relative pose")
        if LIVE_PLOT
        else None
    )
    absolute_live_plot = None

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
        absolute_target_xyz, absolute_target_rpy = build_absolute_targets(
            actions,
            initial_position,
            initial_rotation,
        )
        if LIVE_PLOT:
            absolute_live_plot = LiveReplayPlot(
                absolute_target_xyz,
                absolute_target_rpy,
                dataset.fps,
                title="Live absolute pose",
                target_label="IK target",
            )

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
            actual_absolute_xyz.append(
                np.array(
                    [measured["x.pos"], measured["y.pos"], measured["z.pos"]],
                    dtype=np.float64,
                )
            )
            actual_absolute_rpy.append(
                np.array(
                    [measured["roll.pos"], measured["pitch.pos"], measured["yaw.pos"]],
                    dtype=np.float64,
                )
            )

            if live_plot is not None and (
                frame_index % LIVE_PLOT_EVERY_N_FRAMES == 0
                or frame_index == len(actions) - 1
            ):
                live_plot.update(actual_xyz, actual_rpy)
                if absolute_live_plot is not None:
                    absolute_live_plot.update(actual_absolute_xyz, actual_absolute_rpy)

        print("\nReplay finished.")
    except KeyboardInterrupt:
        print("\nReplay stopped.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        if live_plot is not None:
            live_plot.close()
        if absolute_live_plot is not None:
            absolute_live_plot.close()

    if actual_xyz:
        plot_trajectories(dataset_xyz, np.stack(actual_xyz), dataset.fps)
        plot_orientations(dataset_rpy, np.stack(actual_rpy), dataset.fps)
        plot_absolute_pose(
            absolute_target_xyz,
            np.stack(actual_absolute_xyz),
            absolute_target_rpy,
            np.stack(actual_absolute_rpy),
            dataset.fps,
        )
        plt.show()


if __name__ == "__main__":
    main()
