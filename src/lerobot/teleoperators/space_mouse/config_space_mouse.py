import numpy as np
from dataclasses import dataclass

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("space_mouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    max_speed: float = 0.05
    max_rot_speed: float = 0.1
    deadzone: float = 200.0
    alpha: float = 0.3
    poll_rate: int = 100
    device_num: int = 0
    x_init: float = 0.5
    y_init: float = 0.5
    z_init: float = 0.5
    roll_init: float = np.pi
    pitch_init: float = 0.0
    yaw_init: float = 0.0