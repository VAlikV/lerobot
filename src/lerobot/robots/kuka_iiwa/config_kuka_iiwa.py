from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

DEFAULT_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
)

@RobotConfig.register_subclass("kuka_iiwa")
@dataclass
class KukaIiwaConfig(RobotConfig):
    """Configuration class for KUKA iiwa robots."""

    urdf_path: str = "robots/iiwa2_gripper.urdf"
    use_task_space: bool = True
    # Requires use_task_space=False. Joint observations/actions use absolute degrees.
    use_direct_joint_control: bool = False

    joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_NAMES))

    gripper_port: str = "/dev/ttyUSB0"
    gripper_baudrate: int = 115200

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    resolution: tuple = (224, 224)
    camera_timeout_ms: int = 5000

    # limits: tuple = ((-0.1, 0.1), (-0.1, 0.1), (0.03, 0.1))

    action_pos_scale: int = 1000
    action_angle_scale: int = 100

    # TCP: [x, y, z, roll, pitch, yaw]; joint mode: seven angles in degrees.
    # Both layouts accept an optional final gripper position.
    reset_pose: list[float] | None = None
    reset_time_s: float = 3.0
    reset_fps: int = 30

    control_hz: int = 100

    def __post_init__(self):
        super().__post_init__()
        if self.use_direct_joint_control and self.use_task_space:
            raise ValueError("Direct joint control requires use_task_space=False.")
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7 or "gripper" in self.joint_names:
            raise ValueError("joint_names must contain seven unique arm joint names, excluding gripper.")
