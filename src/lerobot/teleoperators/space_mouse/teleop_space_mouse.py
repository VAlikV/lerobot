import logging

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from .config_space_mouse import SpaceMouseTeleopConfig

logger = logging.getLogger(__name__)


class SpaceMouseTeleop(Teleoperator):
    config_class = SpaceMouseTeleopConfig
    name = "space_mouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._sm = None

    @property
    def action_features(self) -> dict:
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
    def feedback_features(self) -> dict:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return self._sm is not None
    
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
        
        from rc10_api.space_mouse import SpaceMouseWrapper

        self._sm = SpaceMouseWrapper(
            max_speed=self.config.max_speed,
            max_rot_speed=self.config.max_rot_speed,
            deadzone=self.config.deadzone,
            alpha=self.config.alpha,
            poll_rate=self.config.poll_rate,
            device_num=self.config.device_num,
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
        x, y, z, roll, pitch, yaw = self._sm.get_delta()
        gripper = self._sm.get_gripper_state()
        return {
            "x.delta": float(x)*self.config.action_pos_scale,
            "y.delta": float(y)*self.config.action_pos_scale,
            "z.delta": float(z)*self.config.action_pos_scale,
            "roll.delta": float(roll)*self.config.action_angle_scale,
            "pitch.delta": float(pitch)*self.config.action_angle_scale,
            "yaw.delta": float(yaw)*self.config.action_angle_scale,
            "gripper.pos": float(gripper),
        }

    def send_feedback(self, feedback: dict) -> None:
        pass
    
    def disconnect(self):
        if self._sm is not None:
            self._sm.stop()
            self._sm = None
        logger.info(f"{self} disconnected")

# ==============================================================================

class SpaceMouseTeleopCut(Teleoperator):
    config_class = SpaceMouseTeleopConfig
    name = "space_mouse_cut"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._sm = None

    @property
    def action_features(self) -> dict:
        return {
            "x.delta": float,
            "y.delta": float,
            "z.delta": float,
            # "roll.delta": float,
            # "pitch.delta": float,
            "yaw.delta": float,
            "gripper.pos": float,
        }
    
    @property
    def feedback_features(self) -> dict:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return self._sm is not None
    
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
        
        from rc10_api.space_mouse import SpaceMouseWrapper

        self._sm = SpaceMouseWrapper(
            max_speed=self.config.max_speed,
            max_rot_speed=self.config.max_rot_speed,
            deadzone=self.config.deadzone,
            alpha=self.config.alpha,
            poll_rate=self.config.poll_rate,
            device_num=self.config.device_num,
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
        x, y, z, roll, pitch, yaw = self._sm.get_delta()
        gripper = self._sm.get_gripper_state()
        return {
            "x.delta": float(x)*self.config.action_pos_scale,
            "y.delta": float(y)*self.config.action_pos_scale,
            "z.delta": float(z)*self.config.action_pos_scale,
            # "roll.delta": float(roll),
            # "pitch.delta": float(pitch),
            "yaw.delta": float(yaw)*self.config.action_angle_scale,
            "gripper.pos": float(gripper),
        }

    def send_feedback(self, feedback: dict) -> None:
        pass
    
    def disconnect(self):
        if self._sm is not None:
            self._sm.stop()
            self._sm = None
        logger.info(f"{self} disconnected")


