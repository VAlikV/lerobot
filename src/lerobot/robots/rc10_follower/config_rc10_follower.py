from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("rc10_follower")
@dataclass
class RC10FollowerConfig(RobotConfig):
    """Configuration class for RC10 Follower robots."""

    ip: str = "10.10.10.10"
    rate_hz: int = 100
    velocity: float = 1.0   # m/s
    acceleration: float = 1.0   # m/s^2
    threshold_position: float = 0.001   # meter
    threshold_angle: float = 1.0    # degree
    gripper_port: str = "/dev/ttyUSB0"
    gripper_baudrate: int = 115200
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    resolution: tuple = (224, 224)
    limits: tuple = ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1))

    action_pos_scale: int = 1000
    action_angle_scale: int = 100


