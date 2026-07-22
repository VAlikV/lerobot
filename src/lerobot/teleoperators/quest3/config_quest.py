from dataclasses import dataclass

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

    # Коэффициенты масштабирования приращений.
    position_scale: float = 1.0
    rotation_scale: float = 1.0