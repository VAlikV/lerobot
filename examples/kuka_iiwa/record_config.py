from dataclasses import dataclass

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import CameraConfig
from lerobot.robots.kuka_iiwa import KukaIiwaConfig
from lerobot.teleoperators.kuka_leader import KukaLeaderConfig


@dataclass
class PipelineConfig:
    joint_offset_deg: float
    gripper_threshold: float
    gripper_reverse: bool


@dataclass
class DatasetConfig:
    repo_id: str
    task: str

    resume: bool
    root: str | None

    num_episodes: int
    episode_time_s: float

    reset_to_pose: bool
    reset_pose: list[float]
    reset_time_s: float

    use_tts: bool


@dataclass
class RecordingConfig:
    fps: int
    robot: KukaIiwaConfig
    leader: KukaLeaderConfig
    pipeline: PipelineConfig
    dataset: DatasetConfig
