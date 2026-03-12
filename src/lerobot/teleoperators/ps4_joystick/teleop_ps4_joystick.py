import logging

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
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
        x, y, z, roll, pitch, yaw = self._ps4_joystick.get_joystick()
        gripper = self._ps4_joystick.get_gripper_state()

        return {
            "x.pos": float(x),
            "y.pos": float(y),
            "z.pos": float(z),
            "yaw.pos": float(yaw),
            "gripper.pos": float(gripper),
        }
    
    def send_feedback(self, feedback: dict) -> None:
        pass
    
    def disconnect(self):
        if self._ps4_joystick is not None:
            self._ps4_joystick.stop()
            self._ps4_joystick = None
        logger.info(f"{self} disconnected")