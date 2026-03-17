import logging
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_ps4_joystick import PS4JoystickTeleopConfig

logger = logging.getLogger(__name__)

class PS4JoystickTeleop(Teleoperator):
    config_class = PS4JoystickTeleopConfig
    name = "ps4_joystick"
    def __init__(self, config: PS4JoystickTeleopConfig):
        super().__init__(config)
        self.config = config
        self._ps4_joystick = None

    @property
    def action_features(self) -> dict:
        if self.config.delta_mode:
            return {
                "dtype": "float32",
                "shape": (5,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_yaw": 3, "gripper": 4},
            }
        return {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            "yaw.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._ps4_joystick is not None

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

        from rc10_api.ps4_joystick import PS4Joystick

        self._ps4_joystick = PS4Joystick(
            max_speed=self.config.max_speed,
            max_rot_speed=self.config.max_rot_speed,
            deadzone=self.config.deadzone,
            alpha=self.config.alpha,
            poll_rate=self.config.poll_rate,
            x_init=self.config.x_init,
            y_init=self.config.y_init,
            z_init=self.config.z_init,
            roll_init=self.config.roll_init,
            pitch_init=self.config.pitch_init,
            yaw_init=self.config.yaw_init,
        )
        logger.info(f"{self} connected")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self.config.delta_mode:
            # For HIL-SERL mode return normalized [-1, 1] stikc deflections as np array
            # return np array so InterventionActionProcessorStep uses it directly

            dx, dy, dz, dyaw = self._ps4_joystick.get_normalized_deltas()
            gripper = self._ps4_joystick.get_gripper_state()
            return np.array([dx, dy, dz, dyaw, gripper], dtype=np.float32)

        # this is for ACT recording
        x, y, z, roll, pitch, yaw = self._ps4_joystick.get_joystick()
        gripper = self._ps4_joystick.get_gripper_state()
        return {
            "x.pos": float(x),
            "y.pos": float(y),
            "z.pos": float(z),
            "yaw.pos": float(yaw),
            "gripper.pos": float(gripper),
        }

    @check_if_not_connected
    def get_teleop_events(self) -> dict[str, Any]:
        """Returns HIL-SERL episode control signals from ps4 buttons
        Hold R1=intervention, Triangle=success, Circle=terminate, Square=rerecord"""
        states = self._ps4_joystick.get_button_states()
        return {
            TeleopEvents.IS_INTERVENTION: states["is_intervention"],
            TeleopEvents.SUCCESS: states["success"],
            TeleopEvents.TERMINATE_EPISODE: states["terminate_episode"],
            TeleopEvents.RERECORD_EPISODE: states["rerecord_episode"],
        }

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self):
        if self._ps4_joystick is not None:
            self._ps4_joystick.stop()
            self._ps4_joystick = None
        logger.info(f"{self} disconnected")
