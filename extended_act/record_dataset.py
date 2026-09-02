from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.extended_act.configuration_act import OBS_GEOMETRY
from lerobot.policies.extended_act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_processors, make_robot_env
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

try:
    from .data_extractor import DataExtractor, feature_aggregation
except ImportError:
    from data_extractor import DataExtractor, feature_aggregation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- user-tunable -------------------------------------------------------------
CONFIG_PATH = "kuka/configs/kuka_device_assemble.json"

# Set POLICY_DIR to None for pure human recording. When set, the policy drives
# the robot until the gamepad intervention button is held.
POLICY_DIR: str | None = None
POLICY_DATASET_REPO_ID: str | None = None

POLICY_DIR: str | None = "outputs/device_assemble2/extended_act_abs_stage3"
POLICY_DATASET_REPO_ID: str | None = "local/kuka_device_assemble2_abs_stage3_geometry"
POLICY_DATASET_ROOT: Path | None = None
POLICY_DEVICE = "cuda"
SEGMENTATION_MODEL_PATH = Path("extended_act/models/nano_assemble.pt")
PERCEPTION_DEVICE: str | None = None

REPO_ID = "local/kuka_test_2"
# REPO_ID = "local/kuka_device_assemble2_abs_stage3_part3"
TASK_DESCRIPTION = "kuka_assemble"
NUM_EPISODES = 20
EPISODE_TIME_S = 60
FPS = 30

N_ACTION_STEPS = 30

USE_TTS = True

SHOW_FORCE_VECTOR = True
FORCE_VECTOR_SCALE_N = 50.0
FORCE_WINDOW_SIZE = 600
# ----------------------------------------------------------------------ж------


KUKA_RELATIVE_ACTION_NAMES = [
    "x.rel",
    "y.rel",
    "z.rel",
    "roll.rel",
    "pitch.rel",
    "yaw.rel",
    "gripper.pos",
]


class OnlineGeometryExtractor:
    """Build the geometry tensor expected by Extended ACT from live cameras."""

    def __init__(
        self,
        *,
        metadata: LeRobotDatasetMetadata,
        expected_shape: tuple[int, ...],
        model_path: Path,
        device: str | None,
    ) -> None:
        extraction_info = metadata.info.get("geometry_extraction")
        if not extraction_info:
            raise ValueError(
                "Policy dataset metadata does not contain `geometry_extraction`. "
                "Use the dataset produced by convert_dataset_with_geometry.py."
            )

        self.camera_keys = tuple(extraction_info["camera_keys"])
        self.classes = tuple(extraction_info["classes"])
        self.feature_names = tuple(extraction_info["feature_names"])
        self.expected_shape = expected_shape
        actual_shape = (
            len(self.camera_keys),
            len(self.classes),
            len(self.feature_names),
        )
        if actual_shape != expected_shape:
            raise ValueError(
                f"Policy expects geometry shape {expected_shape}, but dataset metadata "
                f"describes {actual_shape}."
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"Segmentation model not found: {model_path}")

        self.extractor = DataExtractor(
            model_path=model_path,
            classes=self.classes,
            device=device,
        )
        self._cached_geometry: torch.Tensor | None = None

    @staticmethod
    def _tensor_rgb_to_bgr(image: torch.Tensor) -> np.ndarray:
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    "Online rollout expects one observation at a time, "
                    f"got image batch shape {tuple(image.shape)}."
                )
            image = image.squeeze(0)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected image shape [3, H, W], got {tuple(image.shape)}.")

        image = image.detach().to(device="cpu", dtype=torch.float32)
        rgb = (
            image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
        ).round().astype(np.uint8)
        return rgb[:, :, ::-1].copy()

    @torch.no_grad()
    def extract(self, observation: dict[str, Any]) -> torch.Tensor:
        per_camera: list[np.ndarray] = []
        for camera_key in self.camera_keys:
            if camera_key not in observation:
                raise KeyError(
                    f"Live observation does not contain geometry camera `{camera_key}`."
                )
            image = observation[camera_key]
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"`{camera_key}` must be a torch.Tensor.")

            detections = self.extractor(self._tensor_rgb_to_bgr(image))
            per_camera.append(
                feature_aggregation(detections, self.classes).astype(
                    np.float32,
                    copy=False,
                )
            )

        geometry = np.stack(per_camera, axis=0)
        if geometry.shape != self.expected_shape:
            raise RuntimeError(
                f"Extracted geometry has shape {geometry.shape}, expected {self.expected_shape}."
            )

        # Environment observations are already batched with B=1.
        return torch.from_numpy(geometry).unsqueeze(0)

    def get(self, observation: dict[str, Any], *, refresh: bool) -> torch.Tensor:
        if refresh or self._cached_geometry is None:
            self._cached_geometry = self.extract(observation)
        return self._cached_geometry

    def reset(self) -> None:
        self._cached_geometry = None


@dataclass
class PolicyBundle:
    metadata: LeRobotDatasetMetadata
    policy: ACTPolicy
    preprocess: Any
    postprocess: Any
    device: torch.device
    geometry: OnlineGeometryExtractor


class ForceVectorVisualizer:
    """Realtime isometric view of the KUKA force vector [Fx, Fy, Fz]."""

    WINDOW_NAME = "KUKA force vector"

    def __init__(self, *, scale_n: float, window_size: int) -> None:
        if scale_n <= 0:
            raise ValueError("`FORCE_VECTOR_SCALE_N` must be greater than zero.")
        self.scale_n = float(scale_n)
        self.window_size = int(window_size)
        self.origin = np.array([self.window_size // 2, self.window_size // 2], dtype=np.float32)

        # Isometric projection: columns are the screen directions of X, Y and Z.
        self.projection = np.array(
            [
                [0.866, -0.866, 0.0],
                [0.5, 0.5, -1.0],
            ],
            dtype=np.float32,
        )
        self.axis_length_px = int(self.window_size * 0.32)

    def _project(self, vector: np.ndarray, length_px: float) -> tuple[int, int]:
        point = self.origin + self.projection @ vector * float(length_px)
        return int(round(point[0])), int(round(point[1]))

    def update(self, obs: dict) -> None:
        agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
        if agent_pos.shape[0] < 9:
            raise ValueError(
                "KUKA force visualization expects agent_pos layout "
                "[x, y, z, roll, pitch, yaw, Fx, Fy, Fz, ...]; "
                f"got shape {agent_pos.shape}."
            )

        force = agent_pos[6:9]
        magnitude = float(np.linalg.norm(force))
        canvas = np.full((self.window_size, self.window_size, 3), 245, dtype=np.uint8)
        origin = tuple(self.origin.astype(int))

        axis_colors = [(40, 40, 220), (40, 170, 40), (220, 100, 30)]
        for axis_idx, (label, color) in enumerate(zip(("X", "Y", "Z"), axis_colors, strict=True)):
            direction = np.zeros(3, dtype=np.float32)
            direction[axis_idx] = 1.0
            endpoint = self._project(direction, self.axis_length_px)
            cv2.arrowedLine(canvas, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=0.08)
            cv2.putText(
                canvas,
                label,
                (endpoint[0] + 7, endpoint[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        normalized_force = force / self.scale_n
        force_endpoint = self._project(normalized_force, self.axis_length_px)
        cv2.arrowedLine(canvas, origin, force_endpoint, (20, 20, 20), 5, cv2.LINE_AA, tipLength=0.12)
        cv2.circle(canvas, origin, 5, (20, 20, 20), -1, cv2.LINE_AA)

        lines = (
            f"Fx: {force[0]:+8.2f} N",
            f"Fy: {force[1]:+8.2f} N",
            f"Fz: {force[2]:+8.2f} N",
            f"|F|: {magnitude:8.2f} N",
            f"Scale: {self.scale_n:.1f} N",
        )
        for row, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (20, 35 + row * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self.WINDOW_NAME)


def _validate_dataset_coordinate_mode(mode: str) -> None:
    if mode not in ("absolute", "relative_to_start"):
        raise ValueError(
            "`dataset.action_recording_mode` must be either 'absolute' or "
            f"'relative_to_start'; got {mode!r}"
        )


def _recording_action_features(env, mode: str) -> dict:
    if mode == "absolute":
        return env.get_recording_action_features("absolute")
    return {
        "dtype": "float32",
        "shape": (len(KUKA_RELATIVE_ACTION_NAMES),),
        "names": KUKA_RELATIVE_ACTION_NAMES,
    }


def _angle_diff(angle: np.ndarray | float, origin: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) - np.asarray(origin) + np.pi) % (2.0 * np.pi) - np.pi


def _episode_origin_from_observation(obs: dict) -> np.ndarray:
    origin = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
    if origin.shape[0] < 6:
        raise ValueError(f"KUKA agent_pos must have at least 6 pose values; got shape {origin.shape}")
    return origin


def _make_relative_observation(obs: dict, episode_origin: np.ndarray, mode: str) -> dict:
    if mode == "absolute":
        return obs

    relative_obs = dict(obs)
    agent_pos = np.asarray(relative_obs["agent_pos"], dtype=np.float32).copy()
    agent_pos[:3] -= episode_origin[:3]
    agent_pos[3:6] = _angle_diff(agent_pos[3:6], episode_origin[3:6]).astype(np.float32)
    relative_obs["agent_pos"] = agent_pos
    return relative_obs


def _make_relative_recording_action(action: np.ndarray, episode_origin: np.ndarray) -> np.ndarray:
    relative_action = np.asarray(action, dtype=np.float32).copy()
    relative_action[:3] -= episode_origin[:3]
    relative_action[3:6] = _angle_diff(relative_action[3:6], episode_origin[3:6]).astype(np.float32)
    return relative_action


def _recording_action(env, mode: str, episode_origin: np.ndarray) -> np.ndarray:
    action = env.get_recording_action("absolute")
    if mode == "absolute":
        return action
    return _make_relative_recording_action(action, episode_origin)


def _relative_policy_action_to_absolute(
    action: dict[str, float],
    episode_origin: np.ndarray,
) -> dict[str, float]:
    return {
        "x.pos": float(episode_origin[0] + action.get("x.rel", action.get("x.pos", 0.0))),
        "y.pos": float(episode_origin[1] + action.get("y.rel", action.get("y.pos", 0.0))),
        "z.pos": float(episode_origin[2] + action.get("z.rel", action.get("z.pos", 0.0))),
        "roll.pos": float(episode_origin[3] + action.get("roll.rel", action.get("roll.pos", 0.0))),
        "pitch.pos": float(episode_origin[4] + action.get("pitch.rel", action.get("pitch.pos", 0.0))),
        "yaw.pos": float(episode_origin[5] + action.get("yaw.rel", action.get("yaw.pos", 0.0))),
        "gripper.pos": float(action.get("gripper.pos", episode_origin[-1])),
    }


def _validate_policy_action_names_for_mode(metadata: LeRobotDatasetMetadata, mode: str) -> None:
    if mode != "relative_to_start":
        return

    action_names = list(metadata.features[ACTION]["names"] or [])
    if action_names != KUKA_RELATIVE_ACTION_NAMES:
        raise ValueError(
            "Relative dataset mode expects a policy trained with relative action names "
            f"{KUKA_RELATIVE_ACTION_NAMES}, but loaded policy dataset has {action_names}. "
            "Use a policy trained on relative-to-start data or switch "
            "`dataset.action_recording_mode` to 'absolute'."
        )


def _process_observation(
    env,
    env_processor,
    obs,
    info: dict,
    *,
    episode_origin: np.ndarray,
    mode: str,
) -> dict:
    obs_for_dataset = _make_relative_observation(obs, episode_origin, mode)
    return _processed_observation(env, env_processor, obs_for_dataset, info)


def _build_dataset_features(env, transition: dict, *, use_gripper: bool, mode: str) -> dict:
    features = {
        ACTION: _recording_action_features(env, mode),
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
    }
    if use_gripper:
        features["complementary_info.discrete_penalty"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        }

    for key, value in transition[TransitionKey.OBSERVATION].items():
        if key == OBS_STATE and isinstance(value, torch.Tensor):
            features[key] = {
                "dtype": "float32",
                "shape": tuple(value.squeeze(0).shape),
                "names": None,
            }
        elif "image" in key and isinstance(value, torch.Tensor):
            features[key] = {
                "dtype": "video",
                "shape": tuple(value.squeeze(0).shape),
                "names": ["channels", "height", "width"],
            }
    return features


def _neutral_delta_action(env, *, use_gripper: bool) -> torch.Tensor:
    action = torch.zeros(int(env.action_space.shape[0]), dtype=torch.float32)
    if use_gripper:
        action[-1] = 1.0
    return action


def _teleop_action_to_delta_tensor(
    teleop_action: dict,
    *,
    use_yaw: bool,
    use_gripper: bool,
) -> torch.Tensor:
    action = [
        teleop_action.get("delta_x", 0.0),
        teleop_action.get("delta_y", 0.0),
        teleop_action.get("delta_z", 0.0),
    ]
    if use_yaw:
        action.append(teleop_action.get("delta_yaw", 0.0))
    if use_gripper:
        action.append(teleop_action.get("gripper", 1.0))
    return torch.tensor(action, dtype=torch.float32)


def _initial_gripper_cmd(env) -> int:
    gripper_pos = float(env.robot._get_pose_observation()["gripper.pos"])
    return 1 if gripper_pos > 0 else 0


def _absolute_gripper_to_binary_cmd(gripper_pos: float) -> int:
    return 1 if float(gripper_pos) > 0 else 0


def _apply_binary_gripper_latch(action, latched_cmd: int) -> int:
    gripper_cmd = int(round(float(action[-1])))
    if gripper_cmd == 0:
        latched_cmd = 0
    elif gripper_cmd == 2:
        latched_cmd = 1
    action[-1] = float(latched_cmd)
    return latched_cmd


def _binary_gripper_to_env_command(action) -> None:
    if int(round(float(action[-1]))) == 1:
        action[-1] = 2.0


def _complete_absolute_action(env, action: dict[str, float]) -> dict[str, float]:
    pose = env.robot._get_pose_observation()
    completed = {
        "x.pos": float(action.get("x.pos", pose["x.pos"])),
        "y.pos": float(action.get("y.pos", pose["y.pos"])),
        "z.pos": float(action.get("z.pos", pose["z.pos"])),
        "roll.pos": float(env.config.fixed_roll),
        "pitch.pos": float(env.config.fixed_pitch),
        "yaw.pos": float(action.get("yaw.pos", env.config.fixed_yaw)),
        "gripper.pos": float(action.get("gripper.pos", pose["gripper.pos"])),
    }

    xyz = np.array([completed["x.pos"], completed["y.pos"], completed["z.pos"]], dtype=np.float32)
    xyz = np.clip(xyz, env.ee_min, env.ee_max)
    completed["x.pos"] = float(xyz[0])
    completed["y.pos"] = float(xyz[1])
    completed["z.pos"] = float(xyz[2])

    if env.use_yaw:
        yaw_offset = float(np.clip(completed["yaw.pos"] - env.config.fixed_yaw, env.yaw_min, env.yaw_max))
        completed["yaw.pos"] = float(env.config.fixed_yaw + yaw_offset)

    return completed


def _apply_absolute_action(env, action: dict[str, float]) -> None:
    action = _complete_absolute_action(env, action)
    env.robot.send_action(action)

    env.target_xyz = np.array([action["x.pos"], action["y.pos"], action["z.pos"]], dtype=np.float32)
    if env.use_yaw:
        env.target_yaw = float(action["yaw.pos"] - env.config.fixed_yaw)
    else:
        env.target_yaw = float(action["yaw.pos"])


def _processed_observation(env, env_processor, obs, info: dict) -> dict:
    transition = create_transition(observation=obs, info=info)
    return env_processor(transition)


def _assert_pending_episode_indices(dataset: LeRobotDataset, pending_episode_buffers: list[dict]) -> None:
    for expected_idx, episode_buffer in enumerate(pending_episode_buffers):
        actual_idx = int(episode_buffer["episode_index"])
        if actual_idx != expected_idx:
            raise RuntimeError(
                "Pending episode buffer indices are inconsistent: "
                f"buffer {expected_idx} has episode_index={actual_idx}. "
                "This usually means the script was run before the rerecord buffer-index fix; "
                "discard this recording run and record again."
            )


def _policy_will_predict_new_chunk(policy: ACTPolicy) -> bool:
    if policy.config.temporal_ensemble_coeff is not None:
        return True
    return len(policy._action_queue) == 0


def _load_policy(
    policy_dir: str,
    dataset_repo_id: str,
    requested_device: str,
    dataset_root: Path | None = None,
) -> PolicyBundle:
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = ACTPolicy.from_pretrained(policy_dir)
    if policy.config.geometry_feature is None:
        raise ValueError(
            "Loaded checkpoint does not declare observation.geometry. "
            "Use an Extended ACT checkpoint."
        )
    if N_ACTION_STEPS > policy.config.chunk_size:
        raise ValueError(
            f"N_ACTION_STEPS={N_ACTION_STEPS} exceeds policy chunk_size="
            f"{policy.config.chunk_size}."
        )
    policy.config.n_action_steps = N_ACTION_STEPS
    device = torch.device(requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu")
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_dir,
        dataset_stats=metadata.stats,
    )
    geometry = OnlineGeometryExtractor(
        metadata=metadata,
        expected_shape=policy.config.geometry_feature.shape,
        model_path=SEGMENTATION_MODEL_PATH,
        device=PERCEPTION_DEVICE,
    )
    logger.info(
        "Extended ACT loaded from %s on %s; geometry cameras=%s classes=%s shape=%s",
        policy_dir,
        device,
        geometry.camera_keys,
        geometry.classes,
        geometry.expected_shape,
    )
    return PolicyBundle(
        metadata=metadata,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=device,
        geometry=geometry,
    )


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    action_recording_mode = cfg.dataset.action_recording_mode
    _validate_dataset_coordinate_mode(action_recording_mode)
    if cfg.env.fps != FPS:
        raise ValueError(f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})")

    policy_bundle = None
    device = torch.device("cpu")
    if POLICY_DIR is not None:
        if POLICY_DATASET_REPO_ID is None:
            raise ValueError("Set POLICY_DATASET_REPO_ID when POLICY_DIR is not None.")
        policy_bundle = _load_policy(
            POLICY_DIR,
            POLICY_DATASET_REPO_ID,
            POLICY_DEVICE,
            POLICY_DATASET_ROOT,
        )
        device = policy_bundle.device
        _validate_policy_action_names_for_mode(policy_bundle.metadata, action_recording_mode)

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, str(device))

    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    ik_cfg = cfg.env.processor.inverse_kinematics
    use_yaw = bool(getattr(ik_cfg, "use_yaw", False)) if ik_cfg else False
    neutral_action = _neutral_delta_action(env, use_gripper=use_gripper)
    dt_s = 1.0 / float(FPS)
    max_episode_steps = int(EPISODE_TIME_S * FPS)
    force_visualizer = (
        ForceVectorVisualizer(
            scale_n=FORCE_VECTOR_SCALE_N,
            window_size=FORCE_WINDOW_SIZE,
        )
        if SHOW_FORCE_VECTOR
        else None
    )

    obs, info = env.reset()
    if force_visualizer is not None:
        force_visualizer.update(obs)
    episode_origin = _episode_origin_from_observation(obs)
    env_processor.reset()
    action_processor.reset()
    transition = _process_observation(
        env,
        env_processor,
        obs,
        info,
        episode_origin=episode_origin,
        mode=action_recording_mode,
    )

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=cfg.dataset.root,
        features=_build_dataset_features(
            env,
            transition,
            use_gripper=use_gripper,
            mode=action_recording_mode,
        ),
        robot_type="kuka_iiwa",
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    logger.info(
        "Recording %d DAgger episodes at %d Hz into %s; policy=%s",
        NUM_EPISODES,
        FPS,
        REPO_ID,
        POLICY_DIR or "disabled",
    )
    logger.info("Dataset action recording mode: %s", action_recording_mode)
    logger.info("Hold the gamepad intervention button to override the policy.")
    if USE_TTS:
        log_say(f"Recording episode 1 of {NUM_EPISODES}")
    if policy_bundle is not None:
        policy_bundle.policy.reset()
        policy_bundle.geometry.reset()

    episode_idx = 0
    episode_step = 0
    intervention_steps = 0
    episode_start = time.perf_counter()
    pending_episode_buffers = []
    gripper_cmd = _initial_gripper_cmd(env) if use_gripper else 1
    policy_gripper_cmd = gripper_cmd
    was_intervening = False

    try:
        while episode_idx < NUM_EPISODES:
            step_start = time.perf_counter()

            probe_transition = transition.copy()
            probe_transition[TransitionKey.ACTION] = neutral_action.clone()
            probe_transition[TransitionKey.OBSERVATION] = (
                env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
            )
            action_probe = action_processor(probe_transition)
            info = action_probe[TransitionKey.INFO]
            is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))
            if policy_bundle is None:
                is_intervention = True
                info[TeleopEvents.IS_INTERVENTION] = True
                action_probe[TransitionKey.ACTION] = _teleop_action_to_delta_tensor(
                    teleop_device.get_action(),
                    use_yaw=use_yaw,
                    use_gripper=use_gripper,
                )
            if (
                use_gripper
                and policy_bundle is not None
                and is_intervention
            ):
                if not was_intervening:
                    gripper_cmd = policy_gripper_cmd
                    logger.info("Intervention started; initialized gripper state from policy.")
                gripper_cmd = _apply_binary_gripper_latch(action_probe[TransitionKey.ACTION], gripper_cmd)
                policy_gripper_cmd = gripper_cmd
            elif use_gripper and (is_intervention or policy_bundle is None):
                gripper_cmd = _apply_binary_gripper_latch(action_probe[TransitionKey.ACTION], gripper_cmd)
            if (
                policy_bundle is not None
                and was_intervening
                and not is_intervention
            ):
                policy_bundle.policy.reset()
                policy_bundle.geometry.reset()
                logger.info("Intervention ended; reset policy action chunk.")
            done = bool(action_probe.get(TransitionKey.DONE, False))
            reward = float(action_probe.get(TransitionKey.REWARD, 0.0))
            discrete_penalty = float(
                action_probe[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
            )

            if is_intervention or policy_bundle is None or done:
                if is_intervention:
                    intervention_steps += 1
                if use_gripper:
                    _binary_gripper_to_env_command(action_probe[TransitionKey.ACTION])
                obs, env_reward, env_done, env_truncated, env_info = env.step(action_probe[TransitionKey.ACTION])
                reward += float(env_reward)
                done = done or bool(env_done)
                truncated = bool(env_truncated)
                info.update(env_info)
            else:
                policy = policy_bundle.policy
                policy_obs = {
                    key: value
                    for key, value in transition[TransitionKey.OBSERVATION].items()
                    if key in policy.config.input_features and key != OBS_GEOMETRY
                }
                policy_obs[OBS_GEOMETRY] = policy_bundle.geometry.get(
                    policy_obs,
                    refresh=_policy_will_predict_new_chunk(policy),
                )
                policy_obs = policy_bundle.preprocess(policy_obs)
                with torch.no_grad():
                    action_tensor = policy.select_action(policy_obs)
                action_tensor = policy_bundle.postprocess(action_tensor)
                robot_action = make_robot_action(
                    action_tensor,
                    policy_bundle.metadata.features,
                )
                if action_recording_mode == "relative_to_start":
                    robot_action = _relative_policy_action_to_absolute(robot_action, episode_origin)
                if use_gripper and "gripper.pos" in robot_action:
                    policy_gripper_cmd = _absolute_gripper_to_binary_cmd(robot_action["gripper.pos"])
                _apply_absolute_action(env, robot_action)
                obs = env._get_observation()
                truncated = False

            if force_visualizer is not None:
                force_visualizer.update(obs)

            if episode_step + 1 >= max_episode_steps:
                truncated = True

            transition = _process_observation(
                env,
                env_processor,
                obs,
                info,
                episode_origin=episode_origin,
                mode=action_recording_mode,
            )
            truncated = truncated or bool(transition.get(TransitionKey.TRUNCATED, False))

            frame = {
                ACTION: torch.as_tensor(
                    _recording_action(env, action_recording_mode, episode_origin),
                    dtype=torch.float32,
                ),
                REWARD: np.array([reward], dtype=np.float32),
                DONE: np.array([done or truncated], dtype=bool),
                "task": TASK_DESCRIPTION,
            }
            if use_gripper:
                frame["complementary_info.discrete_penalty"] = np.array(
                    [discrete_penalty],
                    dtype=np.float32,
                )
            for key, value in transition[TransitionKey.OBSERVATION].items():
                if isinstance(value, torch.Tensor) and key in dataset.features:
                    frame[key] = value.squeeze(0).detach().cpu()
            dataset.add_frame(frame)

            episode_step += 1

            if done or truncated:
                ep_time = time.perf_counter() - episode_start
                rerecord = bool(info.get(TeleopEvents.RERECORD_EPISODE, False))
                success = bool(info.get(TeleopEvents.SUCCESS, False))
                if rerecord:
                    logger.info("Re-recording episode %d after %.1fs", episode_idx + 1, ep_time)
                    dataset.clear_episode_buffer()
                    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_idx)
                    if USE_TTS:
                        log_say(f"Re-recording episode {episode_idx + 1}")
                else:
                    logger.info(
                        "Episode %d %s: %d steps, %.1fs, %d intervention steps",
                        episode_idx + 1,
                        "SUCCESS" if success else "DONE",
                        episode_step,
                        ep_time,
                        intervention_steps,
                    )
                    pending_episode_buffers.append(dataset.episode_buffer)
                    episode_idx += 1
                    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_idx)

                if episode_idx >= NUM_EPISODES:
                    break

                obs, info = env.reset()
                if force_visualizer is not None:
                    force_visualizer.update(obs)
                episode_origin = _episode_origin_from_observation(obs)
                env_processor.reset()
                action_processor.reset()
                if policy_bundle is not None:
                    policy_bundle.policy.reset()
                    policy_bundle.geometry.reset()
                transition = _process_observation(
                    env,
                    env_processor,
                    obs,
                    info,
                    episode_origin=episode_origin,
                    mode=action_recording_mode,
                )
                episode_step = 0
                intervention_steps = 0
                episode_start = time.perf_counter()
                gripper_cmd = _initial_gripper_cmd(env) if use_gripper else 1
                policy_gripper_cmd = gripper_cmd
                was_intervening = False
                if USE_TTS:
                    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            if not (done or truncated):
                was_intervening = is_intervention
            precise_sleep(max(dt_s - (time.perf_counter() - step_start), 0.0))

    except KeyboardInterrupt:
        logger.info("Recording stopped by user.")
    except Exception:
        logger.exception("Recording failed.")
    finally:
        if force_visualizer is not None:
            try:
                force_visualizer.close()
            except cv2.error:
                logger.exception("force visualization window close failed")
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logger.exception("teleop disconnect failed")
        try:
            if pending_episode_buffers and USE_TTS:
                log_say("Robot disconnected. Saving recorded episodes")
            _assert_pending_episode_indices(dataset, pending_episode_buffers)
            for idx, episode_buffer in enumerate(pending_episode_buffers, start=1):
                logger.info(
                    "Saving episode %d of %d after robot disconnect",
                    idx,
                    len(pending_episode_buffers),
                )
                if USE_TTS:
                    log_say(f"Saving episode {idx} of {len(pending_episode_buffers)}")
                dataset.save_episode(episode_data=episode_buffer)
            dataset.finalize()
            logger.info("Dataset finalized -> %s", REPO_ID)
        except Exception:
            logger.exception("dataset save/finalize failed")


if __name__ == "__main__":
    main()
