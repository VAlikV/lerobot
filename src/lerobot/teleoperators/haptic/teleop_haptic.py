import logging
import socket
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from .config_haptic import HapticTeleopConfig

logger = logging.getLogger(__name__)


class HapticTeleop(Teleoperator):
    config_class = HapticTeleopConfig
    name = "haptic"

    def __init__(self, config: HapticTeleopConfig):
        super().__init__(config)
        self.config = config
        self._haptic = None
        self._position, rot = self._initial_position()
        self._rotation = Rotation.from_euler("xyz", rot, False).as_matrix()
        self._gripper_pos = 1.0

    def _initial_position(self) -> np.ndarray:
        if self.config.init_values is None:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        pos_init_values = np.asarray(self.config.init_values[:3], dtype=np.float64)
        rot_init_values = np.asarray(self.config.init_values[3:6], dtype=np.float64)

        return pos_init_values.copy(), rot_init_values.copy()

    @property
    def action_features(self) -> dict:
        if self.config.delta_mode:
            return {
                "x.delta": float,
                "y.delta": float,
                "z.delta": float,
                "roll.pos": float,
                "pitch.pos": float,
                "yaw.pos": float,
                "gripper.pos": float,
            }
        return {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            "roll.pos": float,
            "pitch.pos": float,
            "yaw.pos": float,
            "gripper.pos": float,
        }

    def _parse_packet(self, data: bytes) -> np.ndarray | None:
        try:
            text = data.decode().strip().strip("[]()")
        except UnicodeDecodeError:
            logger.warning("Ignoring non-text haptic UDP packet")
            return None

        packet = np.fromstring(text, sep=",", dtype=np.float64)
        if packet.size != 12:
            packet = np.fromstring(text, sep=" ", dtype=np.float64)

        if packet.size != 12:
            logger.warning("Ignoring haptic UDP packet with %d values, expected 12", packet.size)
            return None

        return packet

    def _read_latest_packet(self) -> np.ndarray | None:
        latest_packet = None

        while True:
            try:
                data, _ = self._haptic.recvfrom(self.config.recv_buffer_size)
            except (BlockingIOError, socket.timeout):
                break

            packet = self._parse_packet(data)
            if packet is not None:
                latest_packet = packet

        return latest_packet

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._haptic is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")

        self._haptic = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._haptic.bind((self.config.ip, self.config.port))
        self._haptic.setblocking(False)
        logger.info(f"{self} connected")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        packet = self._read_latest_packet()
        if packet is not None:
            self._position += packet[:3]
            self._rotation = packet[3:].reshape(3, 3)

        roll, pitch, yaw = Rotation.from_matrix(self._rotation).as_euler("xyz", degrees=False)

        if self.config.delta_mode:
            dx, dy, dz = packet[:3] if packet is not None else (0.0, 0.0, 0.0)
            return {
                "x.delta": float(dx),
                "y.delta": float(dy),
                "z.delta": float(dz),
                "roll.delta": float(roll),
                "pitch.delta": float(pitch),
                "yaw.delta": float(yaw),
                "gripper.pos": float(self._gripper_pos),
            }

        return {
            "x.pos": float(self._position[0]),
            "y.pos": float(self._position[1]),
            "z.pos": float(self._position[2]),
            "roll.pos": float(roll),
            "pitch.pos": float(pitch),
            "yaw.pos": float(yaw),
            "gripper.pos": float(self._gripper_pos),
        }

    @check_if_not_connected
    def get_teleop_events(self) -> dict[str, Any]:
        return {}

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self):
        if self._haptic is not None:
            self._haptic.close()
            self._haptic = None
        logger.info(f"{self} disconnected")

    def reset(self):
        self._position = self._initial_position()
        self._rotation = np.eye(3, dtype=np.float64)
