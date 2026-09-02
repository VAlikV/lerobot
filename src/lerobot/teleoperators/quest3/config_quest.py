from dataclasses import dataclass
from typing import Literal

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("quest_ros2")
@dataclass(kw_only=True)
class QuestRos2Config(TeleoperatorConfig):
    # Топик с положением контроллера.
    tf_topic: str = "/tf"

    # Топик с кнопками контроллера.
    joy_topic: str = "/quest/joystick"

    # child_frame_id нужного контроллера внутри /tf.
    target_frame: str = "hand_right"

    # Индекс кнопки, разрешающей движение.
    move_button_index: int = 7

    # Индекс кнопки управления гриппером.
    gripper_button_index: int = 5

    # Кнопки из Joy.buttons.
    calibration_button: int = 12
    episode_record_button_index: int = 1
    episode_rerecord_button_index: int = 2

    # Коэффициенты масштабирования приращений.
    position_scale: float = 1.0
    rotation_scale: float = 1.0

    # Коэффициенты low-pass фильтра позы: 1.0 отключает сглаживание,
    # меньшие значения дают более плавное движение, но увеличивают задержку.
    position_filter_alpha: float = 0.25
    rotation_filter_alpha: float = 0.20

    control_mode: Literal["delta", "absolute"] = "delta"

    # Постоянная поза губок относительно контроллера. Кватернион задаётся
    # в порядке [x, y, z, w].
    controller_offset_position: tuple[float, float, float] = (0.13865, 0.03027, -0.16595)
    controller_offset_quaternion: tuple[float, float, float, float] = (
        0.19571384,
        -0.04130347,
        0.97617491,
        -0.08409914
    )
