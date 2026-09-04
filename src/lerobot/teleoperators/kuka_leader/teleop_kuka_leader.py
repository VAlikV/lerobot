"""
Teleoperator for a kinematic clone / leader arm of a Kuka iiwa

Use:
lerobot-calibrate --teleop.type=kuka_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_kuka_leader

For teleoperation use the script examples/kuka_iiwa/kuka_leader_to_kuka_iiwa_teleop.py"
"""

import logging
import time
from typing import Any

import math

import serial
import threading

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

    # TODO
    # _get_joint_position and _get_joint_position, switch in get_action

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

        self._reader = SerialFrameReader(self._serial, num_channels=len(self.joint_names))
        self._reader.start()

        self._reader.wait_for_frame(timeout_s=2.0)

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

    @property
    def is_calibrated(self) -> bool:
        return set(self.joint_names) <= set(self.calibration)

    # software-only calibration
    def calibrate(self) -> None:
        """
        Interactive sweep-and-record calibration:
        1. Move each joint to a neutral "home" position, press Enter -> records homing offset.
        2. Move all joints through their full range, press Enter to stop -> records min/max.

        Populates `self.calibration` with one `MotorCalibration` per joint and saves it to `self.calibration_fpath`.
        """

        if not self.is_connected:
            raise RuntimeError(f"{self} must be connected before calibration.")

        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")

        # Home position
        input("Move the arm to the middle of its range of motion and press ENTER...")
        home_raw = self._reader.latest(max_age_s=self.config.max_frame_age_s)
        print("\nRecorded home positions:")
        for joint, value in zip(self.joint_names, home_raw, strict=True):
            print(f"  {joint:<12}: {value}")

        # Range recording
        print()
        mins, maxes = self.record_ranges_of_motion()

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

    # utils
    def record_ranges_of_motion(self) -> tuple[list[int], list[int]]:
        """Record min/max raw encoder positions while the user moves the arm."""

        if not self.is_connected:
            raise RuntimeError(f"{self} must be connected before recording ranges.")

        positions = self._reader.latest(max_age_s=self.config.max_frame_age_s)

        mins = list(positions)
        maxes = list(positions)

        stop_event = threading.Event()

        def _wait_for_enter():
            input()
            stop_event.set()

        listener = threading.Thread(target=_wait_for_enter, daemon=True)
        listener.start()

        while not stop_event.is_set():
            try:
                positions = self._reader.latest(
                    max_age_s=self.config.max_frame_age_s
                )
            except ConnectionError:
                continue

            mins = [
                min(current_min, position)
                for current_min, position in zip(mins, positions, strict=True)
            ]

            maxes = [
                max(current_max, position)
                for current_max, position in zip(maxes, positions, strict=True)
            ]

            # Clear terminal and redraw table
            print("\033[2J\033[H", end="")

            print("Move all joints through their full range of motion.")
            print("Press ENTER to stop recording.\n")

            print(f"{'JOINT':<12} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
            print("-" * 43)

            for i, joint in enumerate(self.joint_names):
                print(
                    f"{joint:<12} | "
                    f"{mins[i]:>6} | "
                    f"{positions[i]:>6} | "
                    f"{maxes[i]:>6}"
                )

            time.sleep(0.02)

        return mins, maxes