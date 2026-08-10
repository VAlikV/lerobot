"""Plot state and action trajectories from one LeRobot dataset episode."""

from pathlib import Path
import math
import threading

import matplotlib.pyplot as plt
import multiprocess.resource_tracker
import numpy as np

# Compatibility workaround for multiprocess 0.70.18 on CPython 3.12.0.
if not hasattr(threading.RLock(), "_recursion_count"):
    def _skip_incompatible_resource_tracker_finalizer(_resource_tracker: object) -> None:
        pass

    multiprocess.resource_tracker.ResourceTracker.__del__ = _skip_incompatible_resource_tracker_finalizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ID = "local/quest_red_cube_pick_and_place_0.25"
EPISODE_INDEX = 0
OUTPUT_DIR = Path(__file__).parent / "plots"


def feature_names(dataset: LeRobotDataset, key: str, width: int) -> list[str]:
    names = dataset.features[key].get("names")
    if names is None or len(names) != width:
        return [f"{key}[{index}]" for index in range(width)]
    return list(names)


def plot_components(
    values: np.ndarray,
    names: list[str],
    time_s: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    columns = 3
    rows = math.ceil(values.shape[1] / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.5 * rows), squeeze=False)

    for index, axis in enumerate(axes.flat):
        if index >= values.shape[1]:
            axis.set_visible(False)
            continue
        axis.plot(time_s, values[:, index], linewidth=1.2)
        axis.set_title(names[index])
        axis.set_xlabel("Time, s")
        axis.grid(True)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_xyz_trajectory(states: np.ndarray, state_names: list[str], output_path: Path) -> None:
    if not {"x", "y", "z"}.issubset(state_names):
        return

    x_index, y_index, z_index = (state_names.index(name) for name in ("x", "y", "z"))
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(states[:, x_index], states[:, y_index], states[:, z_index], linewidth=2)
    axis.scatter(
        states[0, x_index],
        states[0, y_index],
        states[0, z_index],
        color="green",
        label="start",
    )
    axis.scatter(
        states[-1, x_index],
        states[-1, y_index],
        states[-1, z_index],
        color="red",
        label="end",
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(f"XYZ trajectory, episode {EPISODE_INDEX}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main() -> None:
    dataset = LeRobotDataset(REPO_ID, episodes=[EPISODE_INDEX], download_videos=False)
    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    time_s = np.arange(len(states), dtype=np.float32) / dataset.fps

    state_names = feature_names(dataset, "observation.state", states.shape[1])
    action_names = feature_names(dataset, "action", actions.shape[1])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {REPO_ID}")
    print(f"Episode: {EPISODE_INDEX}, frames: {len(states)}, FPS: {dataset.fps}")
    for name, minimum, maximum in zip(
        action_names,
        actions.min(axis=0),
        actions.max(axis=0),
        strict=True,
    ):
        print(f"{name:>16}: min={minimum:+.6f}, max={maximum:+.6f}")

    prefix = OUTPUT_DIR / f"episode_{EPISODE_INDEX:03d}"
    plot_components(states, state_names, time_s, "Observation state", prefix.with_name(prefix.name + "_state.png"))
    plot_components(actions, action_names, time_s, "Action", prefix.with_name(prefix.name + "_action.png"))
    plot_xyz_trajectory(states, state_names, prefix.with_name(prefix.name + "_xyz.png"))
    plt.show()


if __name__ == "__main__":
    main()
