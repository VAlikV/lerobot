from __future__ import annotations

from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Joy
from tf2_msgs.msg import TFMessage

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_quest import QuestRos2Config


class QuestRos2(Teleoperator):
    config_class = QuestRos2Config
    name = "quest_ros2"

    def __init__(self, config: QuestRos2Config):
        super().__init__(config)

        self.config = config

        self._node: Node | None = None
        self._connected = False

        # Последняя полученная поза контроллера.
        self._position: np.ndarray | None = None
        self._quaternion: np.ndarray | None = None

        # Последнее состояние кнопок.
        self._buttons: list[int] = []

        # Предыдущая поза во время удержания кнопки движения.
        self._previous_position: np.ndarray | None = None
        self._previous_quaternion: np.ndarray | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "x.delta": float,
            "y.delta": float,
            "z.delta": float,
            "roll.delta": float,
            "pitch.delta": float,
            "yaw.delta": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        if not rclpy.ok():
            rclpy.init()

        node_name = f"lerobot_quest_ros2_{self.id or 'controller'}"

        self._node = Node(node_name)

        self._node.create_subscription(
            TFMessage,
            self.config.tf_topic,
            self._tf_callback,
            qos_profile_sensor_data,
        )

        self._node.create_subscription(
            Joy,
            self.config.joy_topic,
            self._joy_callback,
            qos_profile_sensor_data,
        )

        self._connected = True

    def _tf_callback(self, msg: TFMessage) -> None:
        target_frame = self.config.target_frame.lstrip("/")

        for transform in msg.transforms:
            child_frame = transform.child_frame_id.lstrip("/")

            if child_frame != target_frame:
                continue

            translation = transform.transform.translation
            rotation = transform.transform.rotation

            self._position = np.array(
                [
                    translation.x,
                    translation.y,
                    translation.z,
                ],
                dtype=np.float64,
            )

            quaternion = np.array(
                [
                    rotation.x,
                    rotation.y,
                    rotation.z,
                    rotation.w,
                ],
                dtype=np.float64,
            )

            norm = np.linalg.norm(quaternion)

            if norm > 1e-8:
                self._quaternion = quaternion / norm

            return

    def _joy_callback(self, msg: Joy) -> None:
        self._buttons = list(msg.axes)

    def _button_pressed(self, index: int) -> bool:
        if index < 0 or index >= len(self._buttons):
            return False

        return bool(round(self._buttons[index]))

    @staticmethod
    def _zero_action(gripper: float) -> dict[str, float]:
        return {
            "x.delta": 0.0,
            "y.delta": 0.0,
            "z.delta": 0.0,
            "roll.delta": 0.0,
            "pitch.delta": 0.0,
            "yaw.delta": 0.0,
            "gripper.pos": gripper,
        }

    def get_action(self) -> dict[str, Any]:
        if not self._connected or self._node is None:
            raise RuntimeError("Quest ROS 2 teleoperator is not connected")

        # Обрабатываем накопившиеся ROS-сообщения.
        # Глубина очереди небольшая, поэтому берутся свежие значения.
        for _ in range(4):
            rclpy.spin_once(
                self._node,
                timeout_sec=0.0,
            )

        gripper_pressed = self._button_pressed(
            self.config.gripper_button_index
        )

        # 1.0 — гриппер закрыт, 0.0 — открыт.
        gripper = 1.0 if gripper_pressed else 0.0

        move_pressed = self._button_pressed(
            self.config.move_button_index
        )

        # Пока кнопка движения не нажата, все приращения нулевые.
        if not move_pressed:
            self._previous_position = None
            self._previous_quaternion = None

            return self._zero_action(gripper)

        # Поза контроллера ещё не получена.
        if self._position is None or self._quaternion is None:
            return self._zero_action(gripper)

        # Первый цикл после нажатия кнопки:
        # запоминаем исходную позу и не создаём скачок.
        if (
            self._previous_position is None
            or self._previous_quaternion is None
        ):
            self._previous_position = self._position.copy()
            self._previous_quaternion = self._quaternion.copy()

            return self._zero_action(gripper)

        # Линейное перемещение с предыдущего цикла.
        delta_position = (
            self._position - self._previous_position
        )

        delta_position *= self.config.position_scale

        # Относительное изменение ориентации.
        previous_rotation = Rotation.from_quat(
            self._previous_quaternion
        )
        current_rotation = Rotation.from_quat(
            self._quaternion
        )

        delta_rotation = (
            previous_rotation.inv() * current_rotation
        )

        # Вектор поворота_buttons:
        # направление — ось, длина — угол в радианах.
        delta_euler = delta_rotation.as_euler(
                "xyz",
                degrees=False,
            )
        delta_euler *= self.config.rotation_scale

        # Текущая поза становится предыдущей для следующего цикла.
        self._previous_position = self._position.copy()
        self._previous_quaternion = self._quaternion.copy()

        return {
            "x.delta": float(delta_position[0]),
            "y.delta": float(delta_position[1]),
            "z.delta": float(delta_position[2]),
            "roll.delta": float(delta_euler[0]),
            "pitch.delta": float(delta_euler[1]),
            "yaw.delta": float(delta_euler[2]),
            "gripper.pos": gripper,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self._connected:
            return

        if self._node is not None:
            self._node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        self._node = None
        self._connected = False

        self._position = None
        self._quaternion = None
        self._previous_position = None
        self._previous_quaternion = None
        self._buttons = []
