import logging
import threading
import time
from copy import copy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.robot_utils import precise_sleep

from ..robot import Robot
from .config_kuka_iiwa import KukaIiwaConfig

logger = logging.getLogger(__name__)


GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0
KUKA_ABSOLUTE_ACTION_NAMES = [
    "x.pos",
    "y.pos",
    "z.pos",
    "roll.pos",
    "pitch.pos",
    "yaw.pos",
    "gripper.pos",
]


class KukaIiwa(Robot):
    config_class = KukaIiwaConfig
    name = "kuka_iiwa_follower"

    def __init__(self, config: KukaIiwaConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._controller = None
        self._gripper = None
        self._home_action = None
        self._target_position = None
        self._target_rotation = None
        self._target_lock = threading.Lock()
        self._controller_lock = threading.Lock()
        self._control_stop_event = threading.Event()
        self._control_thread = None

    @cached_property
    def observation_features(self) -> dict:
        features = {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            "roll.pos": float,
            "pitch.pos": float,
            "yaw.pos": float,
            "force.x": float,
            "force.y": float,
            "force.z": float,
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
        from .gripper import Gripper

        self._controller = fri.KukaController(
            fri.ControlMode.JOINT_POSITION,
            self.config.urdf_path,
            self.config.use_task_space,
        )
        self._controller.start()

        self._gripper = Gripper(
            device=self.config.gripper_port,
            baudrate=self.config.gripper_baudrate,
        )

        for cam in self.cameras.values():
            time.sleep(0.5)
            cam.connect()

        self._home_action = self._make_home_action()
        self._set_target_action(self._home_action)
        self._start_control_thread()
        logger.info(f"{self} connected.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # current_thetta_[0], current_thetta_[1], current_thetta_[2], current_thetta_[3], current_thetta_[4], current_thetta_[5], current_thetta_[6], 
        # current_pos_[0], current_pos_[1], current_pos_[2],
        # current_rot_(0,0), current_rot_(0,1), current_rot_(0,2),
        # current_rot_(1,0), current_rot_(1,1), current_rot_(1,2),
        # current_rot_(2,0), current_rot_(2,1), current_rot_(2,2),
        # force_msg_[0], force_msg_[1], force_msg_[2], force_msg_[3], force_msg_[4], force_msg_[5];

        obs = self._get_pose_observation()
        for cam_name, cam in self.cameras.items():
            image = cam.async_read(timeout_ms=self.config.camera_timeout_ms)
            image = cv2.resize(image, self.config.resolution)
            obs[cam_name] = image
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        # if not self._checkPos(self.tcp[2] + action["z.delta"]/self.config.action_pos_scale):
        #     action["z.delta"] = 0.0

        self._set_target_action(action)
        self._gripper.send(action["gripper.pos"])
        return action

    @check_if_not_connected
    def reset(self) -> RobotAction:
        if self._home_action is None:
            self._home_action = self._make_home_action()

        action = copy(self._home_action)
        if self.config.reset_fps <= 0:
            raise ValueError("`reset_fps` must be greater than 0.")

        dt_s = 1.0 / float(self.config.reset_fps)
        steps = max(1, int(self.config.reset_time_s * self.config.reset_fps))
        for _ in range(steps):
            start_t = time.perf_counter()
            self.send_action(copy(action))
            time.sleep(max(dt_s - (time.perf_counter() - start_t), 0.0))

        logger.info("KUKA reset to start pose.")
        return action

    def disconnect(self) -> None:
        self._stop_control_thread()
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

    def _get_pose_observation(self) -> RobotObservation:
        with self._controller_lock:
            raw_obs = self._controller.get_observation()

        gripper_pos = GRIPPER_OPEN if self._gripper.is_open else GRIPPER_CLOSED

        rot_matrix = np.array([
            raw_obs[10:13],
            raw_obs[13:16],
            raw_obs[16:19],
        ])
        rpy = Rotation.from_matrix(rot_matrix).as_euler("xyz", degrees=False)

        return {
            "x.pos": float(raw_obs[7]),
            "y.pos": float(raw_obs[8]),
            "z.pos": float(raw_obs[9]),
            "roll.pos": float(rpy[0]),
            "pitch.pos": float(rpy[1]),
            "yaw.pos": float(rpy[2]),
            "force.x": float(raw_obs[22]),
            "force.y": float(raw_obs[23]),
            "force.z": float(raw_obs[24]),
            "gripper.pos": float(gripper_pos),
        }

    def _make_home_action(self) -> RobotAction:
        if self.config.reset_pose is None:
            return self._get_pose_observation()

        if len(self.config.reset_pose) not in (6, 7):
            raise ValueError(
                "`reset_pose` must contain [x, y, z, roll, pitch, yaw] "
                "or [x, y, z, roll, pitch, yaw, gripper]."
            )

        gripper_pos = (
            float(self.config.reset_pose[6])
            if len(self.config.reset_pose) == 7
            else GRIPPER_OPEN if self._gripper.is_open else GRIPPER_CLOSED
        )
        return {
            "x.pos": float(self.config.reset_pose[0]),
            "y.pos": float(self.config.reset_pose[1]),
            "z.pos": float(self.config.reset_pose[2]),
            "roll.pos": float(self.config.reset_pose[3]),
            "pitch.pos": float(self.config.reset_pose[4]),
            "yaw.pos": float(self.config.reset_pose[5]),
            "gripper.pos": float(gripper_pos),
        }

    def _set_target_action(self, action: RobotAction) -> None:
        position = np.array([action["x.pos"], action["y.pos"], action["z.pos"]])
        rotation = Rotation.from_euler(
            seq="xyz",
            angles=[action["roll.pos"], action["pitch.pos"], action["yaw.pos"]],
            degrees=False,
        ).as_matrix()

        with self._target_lock:
            self._target_position = position
            self._target_rotation = rotation

    def _start_control_thread(self) -> None:
        if self.config.control_hz <= 0:
            raise ValueError("`control_hz` must be greater than 0.")

        self._control_stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="kuka-iiwa-control",
            daemon=True,
        )
        self._control_thread.start()

    def _stop_control_thread(self) -> None:
        if self._control_thread is None:
            return

        self._control_stop_event.set()
        self._control_thread.join(timeout=2.0)
        if self._control_thread.is_alive():
            logger.warning("KUKA control thread did not stop within timeout.")
        self._control_thread = None

    def _control_loop(self) -> None:
        dt_s = 1.0 / float(self.config.control_hz)
        while not self._control_stop_event.is_set():
            start_t = time.perf_counter()

            with self._target_lock:
                position = None if self._target_position is None else self._target_position.copy()
                rotation = None if self._target_rotation is None else self._target_rotation.copy()

            if position is not None and rotation is not None:
                try:
                    with self._controller_lock:
                        self._controller.set_target(position, rotation)
                except Exception:
                    logger.exception("KUKA control loop failed while sending target.")
                    self._control_stop_event.set()
                    break

            sleep_s = max(dt_s - (time.perf_counter() - start_t), 0.0)
            self._control_stop_event.wait(sleep_s)

    # def _checkPos(self, position):  # noqa: N802
    #     return position > self.config.limits[2][0]


@dataclass
class KukaIiwaRobotEnvConfig:
    """Task-space delta-control parameters for the KUKA iiwa gym environment."""

    ee_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 0.001, "y": 0.001, "z": 0.001}
    )
    ee_bounds_min: list[float] = field(default_factory=lambda: [-0.5, -0.5, 0.05])
    ee_bounds_max: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.7])

    fixed_roll: float = 0.0
    fixed_pitch: float = 0.0
    fixed_yaw: float = 0.0
    home_tcp: list[float] = field(default_factory=lambda: [0.5, 0.0, 0.4, 0.0, 0.0, 0.0])

    reset_time_s: float = 5.0
    reset_fps: int = 30
    use_gripper: bool = True
    use_yaw: bool = False
    randomization_xy: float = 0.0
    randomization_z: float = 0.0


class KukaIiwaRobotEnv(gym.Env):
    """Gym environment for KUKA iiwa with UR10-compatible task-space delta actions.

    Action layout mirrors ``UR10RobotEnv``:
      - ``[dx, dy, dz]``
      - ``[dx, dy, dz, gripper_cmd]`` when ``use_gripper=True``
      - ``[dx, dy, dz, dyaw]`` when ``use_yaw=True``
      - ``[dx, dy, dz, dyaw, gripper_cmd]`` when both are enabled

    The gripper command is discrete: ``0=close``, ``1=stay``, ``2=open``.
    """

    def __init__(self, robot: KukaIiwa, config: KukaIiwaRobotEnvConfig):
        super().__init__()
        self.robot = robot
        self.config = config

        self.ee_step = np.array(
            [
                config.ee_step_sizes["x"],
                config.ee_step_sizes["y"],
                config.ee_step_sizes["z"],
            ],
            dtype=np.float32,
        )
        self.ee_min = np.array(config.ee_bounds_min[:3], dtype=np.float32)
        self.ee_max = np.array(config.ee_bounds_max[:3], dtype=np.float32)

        self.use_gripper = config.use_gripper
        self.use_yaw = config.use_yaw
        self.yaw_step = float(config.ee_step_sizes.get("yaw", 0.01))
        if len(config.ee_bounds_min) >= 4 and len(config.ee_bounds_max) >= 4:
            self.yaw_min = float(config.ee_bounds_min[3])
            self.yaw_max = float(config.ee_bounds_max[3])
        else:
            self.yaw_min, self.yaw_max = -1.5708, 1.5708

        self.current_step = 0
        self.target_xyz: np.ndarray | None = None
        self.target_yaw: float | None = None

        action_dim = 3 + int(self.use_yaw) + int(self.use_gripper)
        low = [-1.0, -1.0, -1.0]
        high = [1.0, 1.0, 1.0]
        if self.use_yaw:
            low.append(-1.0)
            high.append(1.0)
        if self.use_gripper:
            low.append(0.0)
            high.append(2.0)
        self.action_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(action_dim,),
            dtype=np.float32,
        )

        if not self.robot.is_connected:
            self.robot.connect()

        sample = self._get_observation()
        obs_spaces: dict[str, gym.spaces.Box] = {
            OBS_STATE: gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=sample["agent_pos"].shape,
                dtype=np.float32,
            ),
        }
        for cam_name, img in sample["pixels"].items():
            obs_spaces[f"{OBS_IMAGES}.{cam_name}"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=img.shape,
                dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _get_observation(self) -> RobotObservation:
        obs = self.robot.get_observation()
        agent_pos = np.array(
            [
                obs["x.pos"],
                obs["y.pos"],
                obs["z.pos"],
                obs["roll.pos"],
                obs["pitch.pos"],
                obs["yaw.pos"],
                obs["force.x"],
                obs["force.y"],
                obs["force.z"],
                obs["gripper.pos"],
            ],
            dtype=np.float32,
        )
        pixels = {
            cam_name: obs[cam_name]
            for cam_name in self.robot.cameras
            if cam_name in obs
        }
        return {"agent_pos": agent_pos, "pixels": pixels}

    def _send_target(self, xyz: np.ndarray, yaw: float, gripper_cmd: int = 1) -> None:
        target_yaw = float(self.config.fixed_yaw + yaw) if self.use_yaw else float(yaw)
        gripper_pos = self.robot._get_pose_observation()["gripper.pos"]
        if gripper_cmd == 0:
            gripper_pos = GRIPPER_CLOSED
        elif gripper_cmd == 2:
            gripper_pos = GRIPPER_OPEN

        self.robot.send_action(
            {
                "x.pos": float(xyz[0]),
                "y.pos": float(xyz[1]),
                "z.pos": float(xyz[2]),
                "roll.pos": float(self.config.fixed_roll),
                "pitch.pos": float(self.config.fixed_pitch),
                "yaw.pos": target_yaw,
                "gripper.pos": float(gripper_pos),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[RobotObservation, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        home = list(self.config.home_tcp)
        rng = self.np_random
        if self.config.randomization_xy > 0:
            r = self.config.randomization_xy
            home[0] += float(rng.uniform(-r, r))
            home[1] += float(rng.uniform(-r, r))
        if self.config.randomization_z > 0:
            r = self.config.randomization_z
            home[2] += float(rng.uniform(-r, r))

        self.target_xyz = np.clip(np.array(home[:3], dtype=np.float32), self.ee_min, self.ee_max)
        self.target_yaw = 0.0 if self.use_yaw else float(self.config.fixed_yaw)

        reset_fps = max(1, int(self.config.reset_fps))
        dt_s = 1.0 / float(reset_fps)
        steps = max(1, int(self.config.reset_time_s * reset_fps))
        for _ in range(steps):
            start_t = time.perf_counter()
            self._send_target(self.target_xyz, self.target_yaw, gripper_cmd=2)
            precise_sleep(max(dt_s - (time.perf_counter() - start_t), 0.0))

        self.current_step = 0
        return self._get_observation(), {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 3:
            raise ValueError(f"KukaIiwaRobotEnv action must have at least 3 values; got {action.shape}")

        if self.target_xyz is None:
            obs = self.robot._get_pose_observation()
            self.target_xyz = np.array([obs["x.pos"], obs["y.pos"], obs["z.pos"]], dtype=np.float32)
        if self.target_yaw is None:
            self.target_yaw = 0.0 if self.use_yaw else float(self.config.fixed_yaw)

        self.target_xyz = np.clip(self.target_xyz + action[:3] * self.ee_step, self.ee_min, self.ee_max)
        if self.use_yaw:
            self.target_yaw = float(
                np.clip(self.target_yaw + float(action[3]) * self.yaw_step, self.yaw_min, self.yaw_max)
            )

        gripper_cmd = 1
        if self.use_gripper and action.shape[0] >= 3 + int(self.use_yaw) + 1:
            gripper_cmd = int(round(float(action[-1])))

        self._send_target(self.target_xyz, self.target_yaw, gripper_cmd=gripper_cmd)

        obs = self._get_observation()
        self.current_step += 1
        return obs, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False}

    def get_recording_action_features(self, mode: str = "delta") -> dict:
        if mode != "absolute":
            return {"dtype": "float32", "shape": self.action_space.shape, "names": None}
        return {
            "dtype": "float32",
            "shape": (len(KUKA_ABSOLUTE_ACTION_NAMES),),
            "names": KUKA_ABSOLUTE_ACTION_NAMES,
        }

    def get_recording_action(self, mode: str = "delta") -> np.ndarray:
        if mode != "absolute":
            raise ValueError(f"KukaIiwaRobotEnv only exposes explicit recording action for absolute mode; got {mode!r}")

        if self.target_xyz is None:
            obs = self.robot._get_pose_observation()
            xyz = np.array([obs["x.pos"], obs["y.pos"], obs["z.pos"]], dtype=np.float32)
        else:
            xyz = self.target_xyz.astype(np.float32, copy=True)

        if self.target_yaw is None:
            yaw = float(self.config.fixed_yaw)
        elif self.use_yaw:
            yaw = float(self.config.fixed_yaw + self.target_yaw)
        else:
            yaw = float(self.target_yaw)

        gripper_pos = float(self.robot._get_pose_observation()["gripper.pos"])
        return np.array(
            [
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                float(self.config.fixed_roll),
                float(self.config.fixed_pitch),
                yaw,
                gripper_pos,
            ],
            dtype=np.float32,
        )

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()
