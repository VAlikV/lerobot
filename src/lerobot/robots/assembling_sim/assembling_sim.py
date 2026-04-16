import logging
from functools import cached_property

import cv2
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected

from ..robot import Robot
from .config_assembling_sim import AssemblingSimConfig

from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class AssemblingSim(Robot):
    config_class = AssemblingSimConfig
    name = "assembling_sim"

    def __init__(self, config: AssemblingSimConfig):
        super().__init__(config)
        self._config = config
        self._cameras = config.camera_names
        self._sim = None

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
        for cam_name in self._cameras:
            features[cam_name] = (self._config.resolution[0], self._config.resolution[1], 3)
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
        return self._sim is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("Sim is already started.")

        from simulator_for_il_rl.env import AssemblingEnv

        self._sim = AssemblingEnv(
            xml_path=self._config.xml_path,
            sim_timestep=self._config.sim_timestep,
            control_hz=self._config.control_hz,
            mode=self._config.mode,
            max_episode_steps=self._config.max_episode_steps,
            use_task_space=self._config.use_task_space,
            render_mode=self._config.render_mode
        )

        raw_obs, _ = self._sim.reset()
        self.obs = self._convert_obs(raw_obs)

        logger.info(f"{self} started.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:

        return self.obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:

        raw_action = np.zeros(8)

        raw_action[0] = action["x.pos"]
        raw_action[1] = action["y.pos"]
        raw_action[2] = action["z.pos"]

        roll = action["roll.pos"]
        pitch = action["pitch.pos"]
        yaw = action["yaw.pos"]

        euler = np.array([roll, pitch, yaw])
        r = Rotation.from_euler('xyz', euler, degrees=False)
        quat_xyzw = r.as_quat()

        raw_action[3] = quat_xyzw[3]
        raw_action[4] = quat_xyzw[0]
        raw_action[5] = quat_xyzw[1]
        raw_action[6] = quat_xyzw[2]

        raw_action[7] = action["gripper.pos"]

        raw_obs = self._sim.step(raw_action)
        self.obs = self._convert_obs(raw_obs)
        return action

    def disconnect(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        logger.info(f"{self} disconnected.")

    def _convert_obs(self, raw_obs):

        quat_xyzw = np.array([raw_obs["state"]["ee_quat"][1], 
                              raw_obs["state"]["ee_quat"][2], 
                              raw_obs["state"]["ee_quat"][3], 
                              raw_obs["state"]["ee_quat"][0]])
        r = Rotation.from_quat(quat_xyzw)
        euler = r.as_euler('xyz', degrees=False)

        obs = {
            "x.pos": float(raw_obs["state"]["ee_pos"][0]),
            "y.pos": float(raw_obs["state"]["ee_pos"][1]),
            "z.pos": float(raw_obs["state"]["ee_pos"][2]),
            "roll.pos": float(euler[0]),
            "pitch.pos": float(euler[1]),
            "yaw.pos": float(euler[2]),
            "gripper.pos": 1.0,
        }

        for cam_name in self._cameras:
            image = raw_obs["images"][cam_name]
            image = cv2.resize(image, self._config.resolution)
            obs[cam_name] = image

        return obs
        
    def reset(self):
        raw_obs, _ = self._sim.reset()
        self.obs = self._convert_obs(raw_obs)

# ==============================================================================


class AssemblingSimCut(Robot):
    config_class = AssemblingSimConfig
    name = "assembling_sim_cut"

    def __init__(self, config: AssemblingSimConfig):
        super().__init__(config)
        self._config = config
        self._cameras = config.camera_names
        self._sim = None

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
        for cam_name in self._cameras:
            features[cam_name] = (self._config.resolution[0], self._config.resolution[1], 3)
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
        return self._sim is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("Sim is already started.")

        from simulator_for_il_rl.env import AssemblingEnv

        self._sim = AssemblingEnv(
            xml_path=self._config.xml_path,
            sim_timestep=self._config.sim_timestep,
            control_hz=self._config.control_hz,
            mode=self._config.mode,
            max_episode_steps=self._config.max_episode_steps,
            use_task_space=self._config.use_task_space,
            render_mode=self._config.render_mode
        )

        raw_obs, _ = self._sim.reset()
        self.obs = self._convert_obs(raw_obs)

        logger.info(f"{self} started.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:

        return self.obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:

        raw_action = np.zeros(8)

        raw_action[0] = action["x.pos"]
        raw_action[1] = action["y.pos"]
        raw_action[2] = action["z.pos"]

        roll = self.obs["roll.pos"] #+ action["roll.delta"]/self._config.action_angle_scale
        pitch = self.obs["pitch.pos"] #+ action["pitch.delta"]/self._config.action_angle_scale
        yaw = action["yaw.pos"]

        euler = np.array([roll, pitch, yaw])
        r = Rotation.from_euler('xyz', euler, degrees=False)
        quat_xyzw = r.as_quat()

        raw_action[3] = quat_xyzw[3]
        raw_action[4] = quat_xyzw[0]
        raw_action[5] = quat_xyzw[1]
        raw_action[6] = quat_xyzw[2]
        
        if action["gripper.pos"] < 0:
            raw_action[7] = 1
        else:
            raw_action[7] = 0

        raw_obs, reward, terminated, truncated, info = self._sim.step(raw_action)
        self.obs = self._convert_obs(raw_obs)
        return action

    def disconnect(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        logger.info(f"{self} disconnected.")

    def _convert_obs(self, raw_obs):

        quat_xyzw = np.array([raw_obs["state"]["ee_quat"][1], 
                              raw_obs["state"]["ee_quat"][2], 
                              raw_obs["state"]["ee_quat"][3], 
                              raw_obs["state"]["ee_quat"][0]])
        r = Rotation.from_quat(quat_xyzw)
        euler = r.as_euler('xyz', degrees=False)

        obs = {
            "x.pos": float(raw_obs["state"]["ee_pos"][0]),
            "y.pos": float(raw_obs["state"]["ee_pos"][1]),
            "z.pos": float(raw_obs["state"]["ee_pos"][2]),
            "roll.pos": float(euler[0]),
            "pitch.pos": float(euler[1]),
            "yaw.pos": float(euler[2]),
            "gripper.pos": 1.0,
        }

        for cam_name in self._cameras:
            image = raw_obs["images"][cam_name]
            image = cv2.resize(image, self._config.resolution)
            obs[cam_name] = image

        return obs
        
    def reset(self):
        raw_obs, _ = self._sim.reset()
        self.obs = self._convert_obs(raw_obs)

# ==============================================================================