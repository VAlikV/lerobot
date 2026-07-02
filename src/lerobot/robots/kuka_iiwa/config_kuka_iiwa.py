from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("kuka_iiwa")
@dataclass
class KukaIiwaConfig(RobotConfig):
    """Configuration class for KUKA iiwa robots."""

    urdf_path: str = "robots/iiwa.urdf"
    use_task_space: bool = True

    gripper_port: str = "/dev/ttyUSB0"
    gripper_baudrate: int = 115200

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    resolution: tuple = (224, 224)
    camera_timeout_ms: int = 5000

    # limits: tuple = ((-0.1, 0.1), (-0.1, 0.1), (0.03, 0.1))

    action_pos_scale: int = 1000
    action_angle_scale: int = 100

    reset_pose: list[float] | None = None
    reset_time_s: float = 3.0
    reset_fps: int = 30

    control_hz: int = 100
