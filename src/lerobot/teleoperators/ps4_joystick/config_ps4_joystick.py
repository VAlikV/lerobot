import numpy as np
from dataclasses import dataclass

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("ps4_joystick")
@dataclass
class PS4JoystickTeleopConfig(TeleoperatorConfig):
    max_speed: float = 0.05
    max_rot_speed: float = 0.1
    deadzone: float = 0.05
    alpha: float = 0.3
    poll_rate: int = 100
    x_init: float = 0.095
    y_init: float = 0.35
    z_init: float = 0.23
    roll_init: float = np.pi
    pitch_init: float = 0.0
    yaw_init: float = 0.0