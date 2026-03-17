from dataclasses import dataclass

import numpy as np

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

    # For HIL-SERL mode: get_action() should return normalized [-1, 1] stick deltas instead of absolute pos. Also enables get_teleop_events()
    delta_mode: bool = False
