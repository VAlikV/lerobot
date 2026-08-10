"""Record synchronized camera-only episodes in LeRobot format."""

from pathlib import Path
import threading
import time

import multiprocess.resource_tracker

# Work around multiprocess 0.70.18 on CPython 3.12.0. LeRobot imports this
# package through Hugging Face datasets even though this script uses threads.
if not hasattr(threading.RLock(), "_recursion_count"):
    def _skip_incompatible_resource_tracker_finalizer(_resource_tracker: object) -> None:
        pass

    multiprocess.resource_tracker.ResourceTracker.__del__ = _skip_incompatible_resource_tracker_finalizer

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep


FPS = 30
WIDTH = 640
HEIGHT = 480
FOURCC = "MJPG"

NUM_EPISODES = 5
EPISODE_DURATION_S = 10
TASK = "Camera recording"

REPO_ID = "local/arm_camera_dataset"
OUTPUT_DIR = Path(__file__).parent / "datasets" / "arm_camera_dataset"

# Change the indices if cameras get different /dev/video* numbers.
CAMERA_INDICES = {
    "front": 0,
}


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")

    cameras = {
        name: OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=index,
                fps=FPS,
                width=WIDTH,
                height=HEIGHT,
                fourcc=FOURCC,
            )
        )
        for name, index in CAMERA_INDICES.items()
    }
    features = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
        for name in cameras
    }
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=OUTPUT_DIR,
        fps=FPS,
        robot_type="camera_only",
        features=features,
        use_videos=True,
        vcodec="auto",
        image_writer_threads=4,
    )

    interrupted = False
    try:
        for name, camera in cameras.items():
            print(f"Connecting camera {name}: {CAMERA_INDICES[name]} ...")
            camera.connect()

        frames_per_episode = round(EPISODE_DURATION_S * FPS)
        for episode_index in range(NUM_EPISODES):
            input(
                f"Press Enter to record episode {episode_index + 1}/{NUM_EPISODES} "
                "(Ctrl+C to stop) ..."
            )

            print("Recording ...")
            for frame_index in range(frames_per_episode):
                cycle_start = time.perf_counter()
                frame = {
                    f"observation.images.{name}": camera.read_latest(max_age_ms=1000)
                    for name, camera in cameras.items()
                }
                frame["task"] = TASK
                dataset.add_frame(frame)

                print(
                    f"\rEpisode {episode_index + 1}/{NUM_EPISODES}: "
                    f"frame {frame_index + 1}/{frames_per_episode}",
                    end="",
                    flush=True,
                )
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - cycle_start), 0.0))

            dataset.save_episode()
            print("\nEpisode saved.")
    except KeyboardInterrupt:
        interrupted = True
        print("\nRecording stopped.")
        if dataset.episode_buffer["size"] > 0:
            dataset.save_episode()
            print("Partial episode saved.")
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        dataset.stop_image_writer()
        dataset.finalize()

    if not interrupted:
        print(f"Dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
