from dataclasses import dataclass, field

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import CameraConfig
from lerobot.robots.kuka_iiwa import KukaIiwaConfig
from lerobot.teleoperators.kuka_leader import KukaLeaderConfig


@dataclass
class PipelineConfig:
    joint_offset_deg: float
    gripper_threshold: float
    gripper_reverse: bool
    scale_factor: float


@dataclass
class DatasetConfig:
    task: str

    root: str | None

    num_episodes: int
    episode_time_s: float

    reset_to_pose: bool
    reset_pose: list[float]
    reset_time_s: float

    use_tts: bool
    repo_id: str | None = None


@dataclass
class RolloutConfig:
    device: str = "cuda"
    max_episodes: int = 5
    max_steps_per_episode: int = 10000
    n_action_steps: int = 30
    temporal_ensemble_coeff: float | None = None
    reset_to_pose: bool = True


@dataclass
class DaggerConfig:
    device: str = "cuda"
    n_action_steps: int = 30
    temporal_ensemble_coeff: float | None = None
    reset_to_pose: bool = True
    max_rewind_s: float = 15.0
    rewind_speed: float = 1.0


@dataclass
class RecordingConfig:
    fps: int
    robot: KukaIiwaConfig
    leader: KukaLeaderConfig
    pipeline: PipelineConfig
    dataset: DatasetConfig
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    dagger: DaggerConfig = field(default_factory=DaggerConfig)
