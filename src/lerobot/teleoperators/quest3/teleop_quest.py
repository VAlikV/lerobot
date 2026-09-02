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
    """ROS 2 Quest controller with a calibrated controller-to-gripper frame."""

    config_class = QuestRos2Config
    name = "quest_ros2"

    def __init__(self, config: QuestRos2Config):
        super().__init__(config)
        self.config = config

        self._offset_position = np.asarray(config.controller_offset_position, dtype=np.float64)
        offset_quaternion = np.asarray(config.controller_offset_quaternion, dtype=np.float64)
        if self._offset_position.shape != (3,):
            raise ValueError("controller_offset_position must contain exactly 3 values")
        if offset_quaternion.shape != (4,):
            raise ValueError(
                "controller_offset_quaternion must contain exactly 4 values in [x, y, z, w] order"
            )
        if not np.all(np.isfinite(self._offset_position)):
            raise ValueError("controller_offset_position must be finite")
        if not np.all(np.isfinite(offset_quaternion)):
            raise ValueError("controller_offset_quaternion must be finite")

        quaternion_norm = np.linalg.norm(offset_quaternion)
        if quaternion_norm <= 1e-8:
            raise ValueError("controller_offset_quaternion cannot be zero")
        self._offset_rotation = Rotation.from_quat(offset_quaternion / quaternion_norm)

        for parameter_name, alpha in (
            ("position_filter_alpha", config.position_filter_alpha),
            ("rotation_filter_alpha", config.rotation_filter_alpha),
        ):
            if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
                raise ValueError(f"{parameter_name} must be in the interval (0, 1]")

        self._node: Node | None = None
        self._connected = False

        # Calibrated gripper pose relative to its pose at calibration time.
        self._position: np.ndarray | None = None
        self._quaternion: np.ndarray | None = None
        self._initial_position: np.ndarray | None = None
        self._initial_quaternion: np.ndarray | None = None

        # Raw TF pose is retained so calibration can be requested from Joy.
        self._raw_position: np.ndarray | None = None
        self._raw_quaternion: np.ndarray | None = None
        self._calibration_requested = False
        self._calibration_button_was_pressed = False

        self._axes: list[float] = []
        self._buttons: list[int] = []
        self._previous_position: np.ndarray | None = None
        self._previous_quaternion: np.ndarray | None = None
        self._last_absolute_action = self._zero_absolute_action(0.0)

    @property
    def action_features(self) -> dict[str, type]:
        pose_suffix = "pos" if self.config.control_mode == "absolute" else "delta"
        return {
            f"x.{pose_suffix}": float,
            f"y.{pose_suffix}": float,
            f"z.{pose_suffix}": float,
            f"roll.{pose_suffix}": float,
            f"pitch.{pose_suffix}": float,
            f"yaw.{pose_suffix}": float,
            "gripper.pos": float,
            "episode_record": bool,
            "episode_rerecord": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._initial_position is not None and self._initial_quaternion is not None

    def calibrate(self) -> None:
        """Use the latest raw controller pose as the coordinate-system origin."""
        if self._raw_position is None or self._raw_quaternion is None:
            self._calibration_requested = True
            return
        self._apply_calibration(self._raw_position, self._raw_quaternion)

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        if self.config.control_mode not in ("delta", "absolute"):
            raise ValueError(
                f"control_mode must be 'delta' or 'absolute', got {self.config.control_mode!r}"
            )

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
            if transform.child_frame_id.lstrip("/") != target_frame:
                continue

            translation = transform.transform.translation
            rotation = transform.transform.rotation
            position = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
            quaternion = np.array(
                [rotation.x, rotation.y, rotation.z, rotation.w],
                dtype=np.float64,
            )
            quaternion_norm = np.linalg.norm(quaternion)
            if quaternion_norm <= 1e-8:
                return
            quaternion /= quaternion_norm

            self._raw_position = position.copy()
            self._raw_quaternion = quaternion.copy()
            if not self.is_calibrated or self._calibration_requested:
                self._apply_calibration(position, quaternion)

            initial_rotation = Rotation.from_quat(self._initial_quaternion)
            current_rotation = Rotation.from_quat(quaternion)

            # Express translation and rotation in the controller's calibrated
            # starting frame rather than in the global TF frame.
            controller_position = initial_rotation.inv().apply(position - self._initial_position)
            controller_rotation = initial_rotation.inv() * current_rotation

            # Transform the relative controller pose into the gripper frame.
            gripper_displacement_in_controller = (
                controller_position
                + controller_rotation.apply(self._offset_position)
                - self._offset_position
            )
            gripper_position = self._offset_rotation.inv().apply(
                gripper_displacement_in_controller
            )
            gripper_rotation = (
                self._offset_rotation.inv() * controller_rotation * self._offset_rotation
            )
            self._update_filtered_pose(gripper_position, gripper_rotation)
            return

    def _update_filtered_pose(
        self,
        position: np.ndarray,
        rotation: Rotation,
    ) -> None:
        """Low-pass filter position and orientation without filtering Euler angles."""
        if self._position is None or self._quaternion is None:
            self._position = position.copy()
            self._quaternion = rotation.as_quat()
            return

        position_alpha = self.config.position_filter_alpha
        self._position += position_alpha * (position - self._position)

        filtered_rotation = Rotation.from_quat(self._quaternion)
        rotation_error = filtered_rotation.inv() * rotation
        filtered_rotation *= Rotation.from_rotvec(
            self.config.rotation_filter_alpha * rotation_error.as_rotvec()
        )
        self._quaternion = filtered_rotation.as_quat()

    def _apply_calibration(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        self._initial_position = position.copy()
        self._initial_quaternion = quaternion.copy()
        self._calibration_requested = False
        self._position = np.zeros(3, dtype=np.float64)
        self._quaternion = Rotation.identity().as_quat()
        self.reset_delta_origin()
        self._last_absolute_action = self._zero_absolute_action(0.0)

    def _joy_callback(self, msg: Joy) -> None:
        self._axes = list(msg.axes)
        self._buttons = list(msg.buttons)

    def _button_pressed(self, index: int) -> bool:
        return 0 <= index < len(self._buttons) and bool(round(self._buttons[index]))

    def _axis_pressed(self, index: int) -> bool:
        return 0 <= index < len(self._axes) and bool(round(self._axes[index]))

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

    @staticmethod
    def _zero_absolute_action(gripper: float) -> dict[str, float]:
        return {
            "x.pos": 0.0,
            "y.pos": 0.0,
            "z.pos": 0.0,
            "roll.pos": 0.0,
            "pitch.pos": 0.0,
            "yaw.pos": 0.0,
            "gripper.pos": gripper,
        }

    @staticmethod
    def _with_episode_buttons(
        action: dict[str, Any],
        record_pressed: bool,
        rerecord_pressed: bool,
    ) -> dict[str, Any]:
        action["episode_record"] = record_pressed
        action["episode_rerecord"] = rerecord_pressed
        return action

    def _absolute_action(self, gripper: float) -> dict[str, float]:
        if self._position is None or self._quaternion is None:
            action = self._last_absolute_action.copy()
            action["gripper.pos"] = gripper
            return action

        position = self._position * self.config.position_scale
        euler = Rotation.from_quat(self._quaternion).as_euler("xyz", degrees=False)
        euler *= self.config.rotation_scale
        return {
            "x.pos": float(position[0]),
            "y.pos": float(position[1]),
            "z.pos": float(position[2]),
            "roll.pos": float(euler[0]),
            "pitch.pos": float(euler[1]),
            "yaw.pos": float(euler[2]),
            "gripper.pos": gripper,
        }

    def get_observation(self) -> dict[str, float]:
        """Return the calibrated absolute gripper pose relative to its start."""
        if not self._connected or self._node is None:
            raise RuntimeError("Quest ROS 2 teleoperator is not connected")
        if self._position is None or self._quaternion is None:
            raise RuntimeError("Quest ROS 2 controller pose has not been received yet")

        position = self._position * self.config.position_scale
        euler = Rotation.from_quat(self._quaternion).as_euler("xyz", degrees=False)
        euler *= self.config.rotation_scale
        return {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "roll": float(euler[0]),
            "pitch": float(euler[1]),
            "yaw": float(euler[2]),
        }

    def reset_delta_origin(self) -> None:
        """Make the next delta action establish a new local action origin."""
        self._previous_position = None
        self._previous_quaternion = None

    def _spin_once(self) -> None:
        if self._node is None:
            return
        for _ in range(4):
            rclpy.spin_once(self._node, timeout_sec=0.0)

    def get_action(self) -> dict[str, Any]:
        if not self._connected or self._node is None:
            raise RuntimeError("Quest ROS 2 teleoperator is not connected")

        self._spin_once()
        gripper = 1.0 if self._axis_pressed(self.config.gripper_button_index) else 0.0
        move_pressed = self._axis_pressed(self.config.move_button_index)

        calibration_pressed = self._button_pressed(self.config.calibration_button)
        calibration_button_pressed = (
            calibration_pressed and not self._calibration_button_was_pressed
        )
        self._calibration_button_was_pressed = calibration_pressed
        if calibration_button_pressed:
            self.calibrate()

        record_pressed = self._button_pressed(self.config.episode_record_button_index)
        rerecord_pressed = self._button_pressed(self.config.episode_rerecord_button_index)

        if self.config.control_mode == "absolute":
            self._last_absolute_action = self._absolute_action(gripper)
            return self._with_episode_buttons(
                self._last_absolute_action.copy(),
                record_pressed,
                rerecord_pressed,
            )

        if not move_pressed:
            self.reset_delta_origin()
            return self._with_episode_buttons(
                self._zero_action(gripper),
                record_pressed,
                rerecord_pressed,
            )
        if self._position is None or self._quaternion is None:
            return self._with_episode_buttons(
                self._zero_action(gripper),
                record_pressed,
                rerecord_pressed,
            )
        if self._previous_position is None or self._previous_quaternion is None:
            self._previous_position = self._position.copy()
            self._previous_quaternion = self._quaternion.copy()
            return self._with_episode_buttons(
                self._zero_action(gripper),
                record_pressed,
                rerecord_pressed,
            )

        delta_position = (self._position - self._previous_position) * self.config.position_scale
        previous_rotation = Rotation.from_quat(self._previous_quaternion)
        current_rotation = Rotation.from_quat(self._quaternion)
        delta_euler = (previous_rotation.inv() * current_rotation).as_euler(
            "xyz", degrees=False
        )
        delta_euler *= self.config.rotation_scale
        self._previous_position = self._position.copy()
        self._previous_quaternion = self._quaternion.copy()

        return self._with_episode_buttons(
            {
                "x.delta": float(delta_position[0]),
                "y.delta": float(delta_position[1]),
                "z.delta": float(delta_position[2]),
                "roll.delta": float(delta_euler[0]),
                "pitch.delta": float(delta_euler[1]),
                "yaw.delta": float(delta_euler[2]),
                "gripper.pos": gripper,
            },
            record_pressed,
            rerecord_pressed,
        )

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
        self._initial_position = None
        self._initial_quaternion = None
        self._raw_position = None
        self._raw_quaternion = None
        self._calibration_requested = False
        self._calibration_button_was_pressed = False
        self.reset_delta_origin()
        self._axes = []
        self._buttons = []
        self._last_absolute_action = self._zero_absolute_action(0.0)
