import logging
from functools import cached_property

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected

from ..robot import Robot
from .config_kuka_iiwa import KukaIiwaConfig

logger = logging.getLogger(__name__)


class KukaIiwa(Robot):
    config_class = KukaIiwaConfig
    name = "kuka_iiwa_follower"

    def __init__(self, config: KukaIiwaConfig):
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
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            "roll.pos": float,
            "pitch.pos": float,
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
            raise RuntimeError("KUKA is already connected.")

        import kuka_fri_py as fri
        # from rc10_api.controller import TaskSpaceJogController
        # from rc10_api.gripper import Gripper

        self._controller = fri.KukaController(
            fri.ControlMode.JOINT_POSITION,
            self.config.urdf_path,
            self.config.use_task_space,
        )
        self._controller.start()

        # self._gripper = Gripper(
        #     device=self.config.gripper_port,
        #     baudrate=self.config.gripper_baudrate,
        # )

        for cam in self.cameras.values():
            cam.connect()

        # self.tcp = self._controller.get_current_tcp()

        logger.info(f"{self} connected.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # current_thetta_[0], current_thetta_[1], current_thetta_[2], current_thetta_[3], current_thetta_[4], current_thetta_[5], current_thetta_[6], 
        # current_pos_[0], current_pos_[1], current_pos_[2],
        # current_rot_(0,0), current_rot_(0,1), current_rot_(0,2),
        # current_rot_(1,0), current_rot_(1,1), current_rot_(1,2),
        # current_rot_(2,0), current_rot_(2,1), current_rot_(2,2),
        # force_msg_[0], force_msg_[1], force_msg_[2], force_msg_[3], force_msg_[4], force_msg_[5];

        raw_obs = self._controller.get_observation()

        rot_matrix = np.array([raw_obs[10:13],
                               raw_obs[13:16],
                               raw_obs[16:19]])
        
        print(rot_matrix)
        
        rpy = Rotation.from_matrix(rot_matrix).as_euler("xyz", degrees=False)
        obs = {
            "x.pos": float(raw_obs[7]),
            "y.pos": float(raw_obs[8]),
            "z.pos": float(raw_obs[9]),
            "roll.pos": float(rpy[0]),
            "pitch.pos": float(rpy[1]),
            "yaw.pos": float(rpy[2]),
            "gripper.pos": 1.0,
        }
        for cam_name, cam in self.cameras.items():
            image = cam.async_read()
            image = cv2.resize(image, self.config.resolution)
            obs[cam_name] = image
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        # if not self._checkPos(self.tcp[2] + action["z.delta"]/self.config.action_pos_scale):
        #     action["z.delta"] = 0.0

        position = np.array([action["x.pos"], action["y.pos"], action["z.pos"]])
        rotation = Rotation.from_euler(
            seq="xyz",
            angles=[action["roll.pos"], action["pitch.pos"], action["yaw.pos"]],
            degrees=False,
        ).as_matrix()

        self._controller.set_target(position, rotation)
        # self._gripper.send(action["gripper.pos"])
        return action

    def disconnect(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller = None
        # if self._gripper is not None:
        #     self._gripper.close()
        #     self._gripper = None
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        logger.info(f"{self} disconnected.")

    # def _checkPos(self, position):  # noqa: N802
    #     return position > self.config.limits[2][0]
