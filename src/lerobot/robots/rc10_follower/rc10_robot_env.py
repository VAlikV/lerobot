"""
Gym env for our RC10 robot with HIL-SERL support

Unlike the standard RobotEnv from src/lerobot/rl/gym_manipulator.py (which depends on bus.motors for servo-based robots),
this environment works directly with our RC10s TaskSpaceJogController in cartesian EE space.

The policy and teleop both operate in normalized [-1, 1] space.
The env scales actions by ee_step_sizes to get actual position changes (meters/radians),
accumulates them into the TCP target, clips to ee_bounds, and sends to the controller.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from lerobot.processor import RobotObservation
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


@dataclass
class RC10EnvConfig:
    """Config for RC10RobotEnv"""
    # Scaling from [-1, 1] policy output to actual end effector movement per step
    ee_step_sizes: dict[str, float] = field(default_factory=lambda: {
        "x": 0.002, # meter per step
        "y": 0.002,
        "z": 0.002,
        "yaw": 0.02,    #radians per step
    })

    ee_bounds: dict[str, list[float]] = field(default_factory=lambda: {
        "min": [-0.0771, 0.2554, 0.2296],
        "max": [0.2836, 0.6417, 0.4079],
    })

    fixed_roll: float = 3.14159265
    fixed_pitch: float = 0.0

    reset_tcp: list[float] | None = None
    reset_time_s: float = 5.0
    display_cameras: bool = False
    use_gripper: bool = True


class RC10RobotEnv(gym.Env):
    """Gym env for our RC10 robot operating in cartesian space

    Action space: [delta_x, delta_y, delta_z, delta_yaw, gripper] in range[-1, 1]
    Observation: {"agent_pos": [x, y, z, yaw, gripper], "pixels": {cam: image}}
    """

    def __init__(self, robot, env_config: RC10EnvConfig):
        super().__init__()
        self.robot = robot
        self.env_config = env_config

        if not self.robot.is_connected:
            self.robot.connect()

        # Current tcp target maintained by env
        self.tcp = list(self.robot._controller.get_current_tcp())

        self.current_step = 0
        self._image_keys = list(self.robot.cameras.keys())

        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces"""
        obs = self._get_observation()

        observation_spaces = {}
        if "pixels" in obs:
            observation_spaces = {
                f"{OBS_IMAGES}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=obs["pixels"][key].shape, dtype=np.uint8
                )
                for key in obs["pixels"]
            }
        observation_spaces[OBS_STATE] = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=obs["agent_pos"].shape,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Action [delta_x, delta_y, delta_z, delta_yaw, gripper] all in range [-1, 1]
        action_dim = 4 # dx, dy, dz, dyaw
        if self.env_config.use_gripper:
            action_dim += 1
        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim),
            high=np.ones(action_dim),
            shape=(action_dim,),
            dtype=np.float32,
        )

    def _get_observation(self) -> RobotObservation:
        """Get current observation: EE pose + gripper + camera images"""
        tcp = self.robot._controller.get_current_tcp()
        gripper_state = 1.0 if self.robot._gripper.is_open else -1.0

        # state [x, y, z, yaw, gripper]
        agent_pos = np.array([
            tcp[0], tcp[1], tcp[2], tcp[5], gripper_state
        ], dtype=np.float32)

        images = {}
        for cam_name, cam in self.robot.cameras.items():
            image = cam.async_read()
            images[cam_name] = image

        return {"agent_pos": agent_pos, "pixels": images}

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """Reset env: move robot to reset_tcp position"""
        super().reset(seed=seed, options=options)

        if self.env_config.reset_tcp is not None:
            logger.info("resetting RC10 to intial tcp position...")
            # print("hello from reset")
            reset_pose = self.env_config.reset_tcp
            self.robot._controller.set_target(
                reset_pose[0], reset_pose[1], reset_pose[2] + 0.05,
                reset_pose[3], reset_pose[4], reset_pose[5],
            )
            precise_sleep(self.env_config.reset_time_s/2)
            self.robot._controller.set_target(
                reset_pose[0], reset_pose[1], reset_pose[2],
                reset_pose[3], reset_pose[4], reset_pose[5],
            )
            precise_sleep(self.env_config.reset_time_s/2)
            # update internal tcp state to match the reset position
            self.tcp = list(self.robot._controller.get_current_tcp())
        else:
            # just read current position as starting point
            self.tcp = list(self.robot._controller.get_current_tcp())

        self.current_step = 0
        obs = self._get_observation()
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """Execute one step scale [-1, 1] action to end effector deltas, accumulate, send

        Args:
            action: numpy array [delta_x, delta_y, delta_z, delta_yaw, (gripper)]
                    all values in [-1, 1]
        """
        step_sizes = self.env_config.ee_step_sizes
        bounds = self.env_config.ee_bounds

        # Scale normalized deltas to meters/radians
        dx = float(action[0]) * step_sizes["x"]
        dy = float(action[1]) * step_sizes["y"]
        dz = float(action[2]) * step_sizes["z"]
        dyaw = float(action[3]) * step_sizes["yaw"]

        # Accumulate into tcp target
        self.tcp[0] += dx
        self.tcp[1] += dy
        self.tcp[2] += dz
        self.tcp[3] = self.env_config.fixed_roll
        self.tcp[4] = self.env_config.fixed_pitch
        self.tcp[5] += dyaw

        # Clip position to workspace bounds
        self.tcp[0] = np.clip(self.tcp[0], bounds["min"][0], bounds["max"][0])
        self.tcp[1] = np.clip(self.tcp[1], bounds["min"][1], bounds["max"][1])
        self.tcp[2] = np.clip(self.tcp[2], bounds["min"][2], bounds["max"][2])

        # Sent to robot
        self.robot._controller.set_target(*self.tcp)


        if self.env_config.use_gripper and len(action) > 4:
            self.robot._gripper.send(float(action[4]))

        obs = self._get_observation()

        if self.env_config.display_cameras:
            self._render(obs)

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, {TeleopEvents.IS_INTERVENTION: False}

    def _render(self, obs: RobotObservation) -> None:
        """Display camera feeds"""
        if "pixels" in obs:
            for key, image in obs["pixels"].items():
                cv2.imshow(key, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Return current tcp as dict (used by step_env_and_process_transition)
            The naming of this function is not accurate, it returns cartesian coordinates not joints"""
        tcp = self.robot._controller.get_current_tcp()
        return {
            "x.pos": float(tcp[0]),
            "y.pos": float(tcp[1]),
            "z.pos": float(tcp[2]),
            "roll.pos": float(tcp[3]),
            "pitch.pos": float(tcp[4]),
            "yaw.pos": float(tcp[5]),
        }

    def close(self) -> None:
        "Disconnect robot"
        if self.robot.is_connected:
            self.robot.disconnect()
