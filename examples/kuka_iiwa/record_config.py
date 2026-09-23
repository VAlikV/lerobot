from dataclasses import dataclass

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import CameraConfig
from lerobot.robots.kuka_iiwa import KukaIiwaConfig
from lerobot.teleoperators.kuka_leader import KukaLeaderConfig


@dataclass
class PipelineConfig:
    joint_offset_deg: float = 2.0
    gripper_threshold: float = 26.0
    gripper_reverse: bool = False


@dataclass
class DatasetConfig:
    repo_id: str = "local/kuka_test_1"
    task: str = "kuka_assemble"

    resume: bool = False
    root: str | None = None

    num_episodes: int = 2
    episode_time_s: float = 60.0

    reset_to_pose: bool = False
    reset_pose: list[float]
    reset_time_s: float = 30.0

    use_tts: bool = True


@dataclass
class RecordingConfig:
    fps: int
    robot: KukaIiwaConfig
    leader: KukaLeaderConfig
    pipeline: PipelineConfig
    dataset: DatasetConfig