"""Convert the handheld Quest parquet episodes to a LeRobot dataset."""

from pathlib import Path
import threading

import cv2
import multiprocess.resource_tracker
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# multiprocess 0.70.18 expects RLock._recursion_count(), which is absent in
# CPython 3.12.0. The converter does not create multiprocess resources, so its
# incompatible shutdown finalizer can safely be disabled in that environment.
if not hasattr(threading.RLock(), "_recursion_count"):
    def _skip_incompatible_resource_tracker_finalizer(_resource_tracker: object) -> None:
        pass

    multiprocess.resource_tracker.ResourceTracker.__del__ = _skip_incompatible_resource_tracker_finalizer


SOURCE_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR = Path(__file__).parent / "quest_red_cube_pick_and_place_0.25"
REPO_ID = "local/quest_red_cube_pick_and_place"
TASK = "Handheld gripper manipulation"
CAMERA_NAME = "handheld"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# EMA: filtered = alpha * current + (1 - alpha) * previous.
# Smaller values smooth more strongly; 1.0 disables smoothing.
SMOOTHING_ALPHA = 0.25

STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper"]


class EpisodeSmoother:
    """Smooth pose deltas and consistently reintegrate the observation pose."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("SMOOTHING_ALPHA must be in (0, 1]")
        self.alpha = alpha
        self.previous_delta = np.zeros(6, dtype=np.float64)
        self.position = np.zeros(3, dtype=np.float64)
        self.rotation = Rotation.identity()

    def update(self, raw_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filtered_delta = self.alpha * raw_delta + (1.0 - self.alpha) * self.previous_delta
        self.previous_delta = filtered_delta

        self.position += filtered_delta[:3]
        self.rotation = self.rotation * Rotation.from_euler(
            "xyz",
            filtered_delta[3:6],
            degrees=False,
        )
        state = np.concatenate(
            [self.position, self.rotation.as_euler("xyz", degrees=False)]
        ).astype(np.float32)
        return filtered_delta.astype(np.float32), state


def parquet_metadata(path: Path) -> dict[str, str]:
    metadata = pq.read_metadata(path).metadata or {}
    return {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in metadata.items()
        if key != b"ARROW:schema"
    }


def decode_image(image_bytes: bytes) -> np.ndarray:
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode a JPEG image")

    source_height, source_width = image_bgr.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if IMAGE_WIDTH < source_width or IMAGE_HEIGHT < source_height
        else cv2.INTER_LINEAR
    )
    image_bgr = cv2.resize(
        image_bgr,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=interpolation,
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def make_frame(row: dict, task: str, smoother: EpisodeSmoother) -> dict:
    action = row["action"]
    delta = action["delta"]
    raw_delta = np.array(
        [
            delta["x"],
            delta["y"],
            delta["z"],
            delta["roll"],
            delta["pitch"],
            delta["yaw"],
        ],
        dtype=np.float64,
    )
    filtered_delta, filtered_state = smoother.update(raw_delta)

    return {
        "observation.state": filtered_state,
        "action": np.concatenate(
            [filtered_delta, np.array([action["gripper_position"]], dtype=np.float32)]
        ),
        f"observation.images.{CAMERA_NAME}": decode_image(row["image"]),
        "task": task,
    }


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if IMAGE_WIDTH <= 0 or IMAGE_HEIGHT <= 0:
        raise ValueError("IMAGE_WIDTH and IMAGE_HEIGHT must be greater than zero")

    episode_paths = sorted(SOURCE_DIR.rglob("*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No parquet episodes found in {SOURCE_DIR}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")

    metadata = parquet_metadata(episode_paths[0])
    fps = int(metadata["fps"])

    image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    for path in episode_paths[1:]:
        episode_fps = int(parquet_metadata(path)["fps"])
        if episode_fps != fps:
            raise ValueError(f"All episodes must have the same FPS: {path} has {episode_fps}, expected {fps}")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
        f"observation.images.{CAMERA_NAME}": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=fps,
        root=OUTPUT_DIR,
        robot_type="quest_gripper",
        features=features,
        use_videos=True,
        vcodec="auto",
    )

    try:
        for episode_number, path in enumerate(episode_paths):
            table = pq.read_table(path)
            print(f"[{episode_number + 1}/{len(episode_paths)}] {path.name}: {table.num_rows} frames")
            smoother = EpisodeSmoother(SMOOTHING_ALPHA)

            for row in table.to_pylist():
                dataset.add_frame(make_frame(row, TASK, smoother))

            dataset.save_episode()
    finally:
        dataset.stop_image_writer()
        dataset.finalize()

    print(f"Converted {len(episode_paths)} episodes to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
