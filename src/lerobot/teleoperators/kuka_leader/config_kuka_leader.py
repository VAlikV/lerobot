from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

DEFAULT_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
)

@TeleoperatorConfig.register_subclass("kuka_leader")
@dataclass
class KukaLeaderConfig(TeleoperatorConfig):
    # Serial port the STM32 board is connected to, e.g. "/dev/ttyACM0"
    port: str

    # Names of the joints
    joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_NAMES))

    baudrate: int = 115_200

    # Whether to use degrees for angles; radians othervise 
    use_degrees: bool = True

    # Seconds to wait after opening the port before trusting incoming data
    boot_delay_s: float = 2.0

    # Max age (seconds) of the latest frame
    max_frame_age_s: float = 0.5

    # TODO
    # Enum: joint, cartesian

    def __post_init__(self):
        if len(self.joint_names) > 8:
            raise ValueError(
                f"KukaLeaderConfig supports at most 8 channels, got {len(self.joint_names)} joint_names."
            )