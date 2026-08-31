"""
Teleoperator for a kinematic clone / leader arm of a Kuka iiwa
"""

import logging
import time
from typing import Any

import math

import serial

from lerobot.motors.motors_bus import MotorCalibration
from lerobot.processor import RobotAction

from ..teleoperator import Teleoperator
from .config_kuka_leader import KukaLeaderConfig
from .kuka_leader_utils import SerialFrameReader

logger = logging.getLogger(__name__)


class KukaLeader(Teleoperator):

    config_class = KukaLeaderConfig
    name = "kuka_leader"

    def __init__(self, config: KukaLeaderConfig):
        super().__init__(config)
        self.config = config
        self.joint_names = list(config.joint_names)
        self._serial: serial.Serial | None = None
        self._reader: SerialFrameReader | None = None


    @property
    def action_features(self) -> dict:
        return {f"{name}.pos": float for name in self.joint_names}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self._serial is not None
            and self._serial.is_open
            and self._reader is not None
            and self._reader.is_alive()
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected.")

        self._serial = serial.Serial(self.config.port, self.config.baudrate, timeout=0.02)
        time.sleep(self.config.boot_delay_s)
        self._serial.reset_input_buffer()

        self._reader = SerialFrameReader(self._serial, num_channels=len(self.joint_names))
        self._reader.start()
        # make sure at least one frame has arrived before we declare success
        self._reader.latest(max_age_s=2.0)

        self.configure()

        if calibrate and not self.is_calibrated:
            self.calibrate()

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        if self._reader is not None:
            self._reader.stop()
            self._reader.join(timeout=1.0)
            self._reader = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

        logger.info(f"{self} disconnected.")

    def configure(self) -> None:
        # Nothing to configure on a passive read-only board.
        pass

    # software-only calibration
    @property
    def is_calibrated(self) -> bool:
        return set(self.joint_names) <= set(self.calibration)

    def calibrate(self) -> None:
        """
        Interactive sweep-and-record calibration:
        1. Move each joint to a neutral "home" position, press Enter -> records homing offset.
        2. Slowly move all joints through their full range, press Enter to stop -> records min/max.

        Populates `self.calibration` with one `MotorCalibration` per joint and saves it to `self.calibration_fpath`.
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} must be connected before calibration.")

        print(f"\nCalibrating {self}.")
        input("Move the arm to the middle of its range of motion and press ENTER...")
        home_raw = self._reader.latest(max_age_s=self.config.max_frame_age_s)

        print("Move every joint through its full range of motion.")
        input("Press Enter to start recording, then move the arm, then press Enter again to stop...")

        mins = list(home_raw)
        maxes = list(home_raw)
        recording = True

        # Simple blocking sweep: sample continuously until the user presses Enter again.
        # (Swap this for a non-blocking key listener if you want live feedback while sweeping.)
        import threading

        stop_event = threading.Event()

        def _wait_for_enter():
            input()
            stop_event.set()

        listener = threading.Thread(target=_wait_for_enter, daemon=True)
        listener.start()

        while not stop_event.is_set():
            try:
                frame = self._reader.latest(max_age_s=self.config.max_frame_age_s)
            except ConnectionError:
                continue
            mins = [min(a, b) for a, b in zip(mins, frame, strict=True)]
            maxes = [max(a, b) for a, b in zip(maxes, frame, strict=True)]
            time.sleep(0.01)

        self.calibration = {}
        for idx, joint in enumerate(self.joint_names):
            self.calibration[joint] = MotorCalibration(
                id=idx,
                drive_mode=0,
                homing_offset=int(home_raw[idx]),
                range_min=int(mins[idx]),
                range_max=int(maxes[idx]),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def _encoder_to_joint_angle(self, joint: str, raw: int) -> float:
        cal = self.calibration.get(joint)
        if cal is None:
            raise RuntimeError(f"No calibration for joint {joint}")
        delta = raw - cal.homing_offset
        angle = delta * (2.0 * math.pi / 4096.0)
        return angle

    # action / feedback

    def get_action(self) -> RobotAction:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        raw_values = self._reader.latest(max_age_s=self.config.max_frame_age_s)
        return {
            f"{joint}.pos": self._encoder_to_joint_angle(joint, raw_values[idx])
            for idx, joint in enumerate(self.joint_names)
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # No actuators on this device -- nothing to send.
        return