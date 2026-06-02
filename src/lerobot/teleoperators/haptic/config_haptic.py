from dataclasses import dataclass

import numpy as np

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("haptic")
@dataclass
class HapticTeleopConfig(TeleoperatorConfig):

    init_values: np.array = None
    ip: str = "127.0.0.1"
    port: int = 8081
    recv_buffer_size: int = 1024

    delta_mode: bool = False
