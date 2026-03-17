import logging
from functools import cached_property

import cv2
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected

from ..robot import Robot
from .config_rc10_follower import RC10FollowerConfig

logger = logging.getLogger(__name__)


class RC10Follower(Robot):
    config_class = RC10FollowerConfig
    name = "rc10_follower"

    def __init__(self, config: RC10FollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._controller = None
        self._gripper = None

    @cached_property
    def observation_features(self) -> dict:
        features = {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            "roll.pos": float,
            "pitch.pos": float,
            "yaw.pos": float,
            "gripper.pos": float,
        }
        for cam_name in self.cameras:
            # cam_cfg = self.config.cameras[cam_name]
            features[cam_name] = (self.config.resolution[0], self.config.resolution[1], 3)
        return features

    @cached_property
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
    def is_connected(self) -> bool:
        return self._controller is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("RC10Follower is already connected.")

        from rc10_api.controller import TaskSpaceJogController
        from rc10_api.gripper import Gripper

        self._controller = TaskSpaceJogController(
            ip=self.config.ip,
            rate_hz=self.config.rate_hz,
            velocity=self.config.velocity,
            acceleration=self.config.acceleration,
            treshold_position=self.config.threshold_position,
            treshold_angel=self.config.threshold_angle,
        )
        self._controller.start()

        self._gripper = Gripper(
            device=self.config.gripper_port,
            baudrate=self.config.gripper_baudrate,
        )

        for cam in self.cameras.values():
            cam.connect()

        self.tcp = self._controller.get_current_tcp()

        logger.info(f"{self} connected.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        tcp = self._controller.get_current_tcp()
        obs = {
            "x.pos": float(tcp[0]),
            "y.pos": float(tcp[1]),
            "z.pos": float(tcp[2]),
            "roll.pos": float(tcp[3]),
            "pitch.pos": float(tcp[4]),
            "yaw.pos": float(tcp[5]),
            "gripper.pos": 1.0 if self._gripper.is_open else -1.0,
        }
        for cam_name, cam in self.cameras.items():
            image = cam.async_read()
            image = cv2.resize(image, self.config.resolution)
            obs[cam_name] = image
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._checkPos(self.tcp[2] + action["z.delta"]/self.config.action_pos_scale):
            action["z.delta"] = 0.0

        self.tcp[0] += action["x.delta"]/self.config.action_pos_scale,
        self.tcp[1] += action["y.delta"]/self.config.action_pos_scale,
        self.tcp[2] += action["z.delta"]/self.config.action_pos_scale,
        self.tcp[3] += action["roll.delta"]/self.config.action_angle_scale,
        self.tcp[4] += action["pitch.delta"]/self.config.action_angle_scale,
        self.tcp[5] += action["yaw.delta"]/self.config.action_angle_scale,

        self._controller.set_target(*self.tcp)
        self._gripper.send(action["gripper.pos"])
        return action

    def disconnect(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller = None
        if self._gripper is not None:
            self._gripper.close()
            self._gripper = None
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        logger.info(f"{self} disconnected.")

    def _checkPos(self, position):
        return position > self.config.limits[2][0]

# ==============================================================================

class RC10FollowerCut(Robot):
    config_class = RC10FollowerConfig
    name = "rc10_follower_cut"

    def __init__(self, config: RC10FollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._controller = None
        self._gripper = None

    @cached_property
    def observation_features(self) -> dict:
        features = {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            # "roll.pos": float,
            # "pitch.pos": float,
            "yaw.pos": float,
            "gripper.pos": float,
        }
        for cam_name in self.cameras:
            # cam_cfg = self.config.cameras[cam_name]
            features[cam_name] = (self.config.resolution[0], self.config.resolution[1], 3)
        return features

    @cached_property
    def action_features(self) -> dict:
        return {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            # "roll.delta": float,
            # "pitch.delta": float,
            "yaw.pos": float,
            "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._controller is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("RC10Follower is already connected.")

        from rc10_api.controller import TaskSpaceJogController
        from rc10_api.gripper import Gripper

        self._controller = TaskSpaceJogController(
            ip=self.config.ip,
            rate_hz=self.config.rate_hz,
            velocity=self.config.velocity,
            acceleration=self.config.acceleration,
            treshold_position=self.config.threshold_position,
            treshold_angel=self.config.threshold_angle,
        )
        self._controller.start()

        self._gripper = Gripper(
            device=self.config.gripper_port,
            baudrate=self.config.gripper_baudrate,
        )

        for cam in self.cameras.values():
            cam.connect()

        self.tcp = self._controller.get_current_tcp()

        logger.info(f"{self} connected.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        tcp = self._controller.get_current_tcp()
        obs = {
            "x.pos": float(tcp[0]),
            "y.pos": float(tcp[1]),
            "z.pos": float(tcp[2]),
            # "roll.pos": float(tcp[3]),
            # "pitch.pos": float(tcp[4]),
            "yaw.pos": float(tcp[5]),
            "gripper.pos": 1.0 if self._gripper.is_open else -1.0,
        }
        for cam_name, cam in self.cameras.items():
            image = cam.async_read()
            image = cv2.resize(image, self.config.resolution)
            obs[cam_name] = image
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:

        if not self._checkPos(action["z.pos"]):
            action["z.pos"] = self.tcp[2]

        print(action)

        self._controller.set_target(
            action["x.pos"],
            action["y.pos"],
            action["z.pos"],
            np.pi,
            0.0,
            action["yaw.pos"],            )
        self._gripper.send(action["gripper.pos"])
        return action

    def disconnect(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller = None
        if self._gripper is not None:
            self._gripper.close()
            self._gripper = None
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        logger.info(f"{self} disconnected.")

    def _checkPos(self, position):  # noqa: N802
        return position > self.config.limits[2][0]

