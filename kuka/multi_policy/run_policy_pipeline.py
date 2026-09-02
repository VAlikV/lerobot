from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    ImageCropResizeProcessorStep,
    Numpy2TorchActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
)
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_robot_env
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POLICIES_CONFIG_PATH = PROJECT_ROOT / "kuka/multi_policy/configs/policies.json"
START_POINTS_PATH = PROJECT_ROOT / "kuka/multi_policy/configs/kuka_policy_start_points.json"
ROBOT_CONFIG_PATH = PROJECT_ROOT / "kuka/act/configs/kuka_device_assemble.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RELATIVE_ACTION_NAMES = [
    "x.rel",
    "y.rel",
    "z.rel",
    "roll.rel",
    "pitch.rel",
    "yaw.rel",
    "gripper.pos",
]
POSE_KEYS = (
    "x.pos",
    "y.pos",
    "z.pos",
    "roll.pos",
    "pitch.pos",
    "yaw.pos",
    "gripper.pos",
)


@dataclass(frozen=True)
class PipelineConfig:
    order: list[str]
    device: str
    fps: int
    max_steps: int
    n_action_steps: int | None
    reset_time_s: float
    use_tts: bool


@dataclass(frozen=True)
class PolicyStage:
    name: str
    start_point: str
    initial_gripper: str
    action_recording_mode: str
    model_dir: str
    dataset_repo_id: str


@dataclass
class LoadedPolicy:
    stage: PolicyStage
    metadata: LeRobotDatasetMetadata
    policy: ACTPolicy
    preprocess: Any
    postprocess: Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def _load_pipeline_config(document: dict[str, Any]) -> PipelineConfig:
    raw = document.get("pipeline")
    if not isinstance(raw, dict):
        raise ValueError(f"{POLICIES_CONFIG_PATH}: missing 'pipeline' object")
    order = raw.get("order")
    if not isinstance(order, list) or not order or not all(isinstance(name, str) for name in order):
        raise ValueError("pipeline.order must be a non-empty list of policy names")
    if len(order) != len(set(order)):
        raise ValueError("pipeline.order contains duplicate policy names")

    n_action_steps = raw.get("n_action_steps")
    if n_action_steps is not None and (not isinstance(n_action_steps, int) or n_action_steps <= 0):
        raise ValueError("pipeline.n_action_steps must be a positive integer or null")
    config = PipelineConfig(
        order=order,
        device=str(raw.get("device", "cuda")),
        fps=int(raw.get("fps", 30)),
        max_steps=int(raw.get("max_steps", 30000)),
        n_action_steps=n_action_steps,
        reset_time_s=float(raw.get("reset_time_s", 5.0)),
        use_tts=bool(raw.get("use_tts", False)),
    )
    if config.fps <= 0 or config.max_steps <= 0 or config.reset_time_s < 0:
        raise ValueError("pipeline fps/max_steps must be positive and reset_time_s cannot be negative")
    return config


def _load_stages(document: dict[str, Any], order: list[str]) -> list[PolicyStage]:
    definitions = document.get("policies")
    if not isinstance(definitions, dict):
        raise ValueError(f"{POLICIES_CONFIG_PATH}: missing 'policies' object")

    stages = []
    for name in order:
        raw = definitions.get(name)
        if not isinstance(raw, dict):
            raise KeyError(f"Pipeline policy {name!r} is not defined")
        inference = raw.get("inference")
        if not isinstance(inference, dict):
            raise ValueError(f"Policy {name!r}: missing 'inference' object")
        model_dir = inference.get("model_dir")
        dataset_repo_id = inference.get("dataset_repo_id")
        if not isinstance(model_dir, str) or not model_dir:
            raise ValueError(f"Policy {name!r}: inference.model_dir must be configured")
        if not isinstance(dataset_repo_id, str) or not dataset_repo_id:
            raise ValueError(f"Policy {name!r}: inference.dataset_repo_id must be configured")

        initial_gripper = raw.get("initial_gripper", "manual")
        mode = raw.get("action_recording_mode", "absolute")
        if initial_gripper not in ("manual", "open", "closed"):
            raise ValueError(f"Policy {name!r}: invalid initial_gripper {initial_gripper!r}")
        if mode not in ("absolute", "relative_to_start"):
            raise ValueError(f"Policy {name!r}: invalid action_recording_mode {mode!r}")
        stages.append(
            PolicyStage(
                name=name,
                start_point=str(raw.get("start_point", name)),
                initial_gripper=initial_gripper,
                action_recording_mode=mode,
                model_dir=model_dir,
                dataset_repo_id=dataset_repo_id,
            )
        )
    return stages


def _load_start_poses(stages: list[PolicyStage]) -> dict[str, list[float]]:
    document = _read_json(START_POINTS_PATH)
    points = document.get("points")
    if not isinstance(points, list):
        raise ValueError(f"{START_POINTS_PATH}: missing 'points' list")

    poses: dict[str, list[float]] = {}
    for stage in stages:
        matches = [point for point in points if isinstance(point, dict) and point.get("name") == stage.start_point]
        if len(matches) != 1:
            raise ValueError(
                f"Policy {stage.name!r}: expected exactly one start point {stage.start_point!r}, "
                f"found {len(matches)}"
            )
        pose = matches[0].get("pose")
        if not isinstance(pose, dict) or any(key not in pose for key in POSE_KEYS):
            raise ValueError(f"Start point {stage.start_point!r} does not contain a complete pose")
        values = [float(pose[key]) for key in POSE_KEYS]
        if stage.initial_gripper == "open":
            values[6] = 1.0
        elif stage.initial_gripper == "closed":
            values[6] = -1.0
        poses[stage.name] = values
    return poses


def _make_env_processor(config: GymManipulatorConfig, device: str):
    steps = [Numpy2TorchActionProcessorStep(), VanillaObservationProcessorStep()]
    image_config = config.env.processor.image_preprocessing
    if image_config is not None:
        steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=image_config.crop_params_dict,
                resize_size=image_config.resize_size,
            )
        )
    steps.extend([AddBatchDimensionProcessorStep(), DeviceProcessorStep(device=device)])
    return DataProcessorPipeline[EnvTransition, EnvTransition](
        steps=steps,
        to_transition=identity_transition,
        to_output=identity_transition,
    )


def _load_policy(stage: PolicyStage, pipeline: PipelineConfig, device: torch.device) -> LoadedPolicy:
    metadata = LeRobotDatasetMetadata(stage.dataset_repo_id)
    action_names = list(metadata.features["action"].get("names") or [])
    if stage.action_recording_mode == "relative_to_start" and action_names != RELATIVE_ACTION_NAMES:
        raise ValueError(
            f"Policy {stage.name!r} is relative but its dataset action names are {action_names}; "
            f"expected {RELATIVE_ACTION_NAMES}"
        )

    policy = ACTPolicy.from_pretrained(stage.model_dir)
    if pipeline.n_action_steps is not None:
        if pipeline.n_action_steps > policy.config.chunk_size:
            raise ValueError(
                f"Policy {stage.name!r}: n_action_steps={pipeline.n_action_steps} exceeds "
                f"chunk_size={policy.config.chunk_size}"
            )
        policy.config.n_action_steps = pipeline.n_action_steps
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=stage.model_dir,
        dataset_stats=metadata.stats,
    )
    policy.reset()
    logger.info("Loaded policy %s from %s", stage.name, stage.model_dir)
    return LoadedPolicy(stage, metadata, policy, preprocess, postprocess)


def _angle_diff(value: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (value - origin + np.pi) % (2.0 * np.pi) - np.pi


def _policy_observation(obs: dict, origin: np.ndarray, mode: str) -> dict:
    if mode == "absolute":
        return obs
    result = dict(obs)
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
    agent_pos[:3] -= origin[:3]
    agent_pos[3:6] = _angle_diff(agent_pos[3:6], origin[3:6]).astype(np.float32)
    result["agent_pos"] = agent_pos
    return result


def _process_observation(env_processor, obs: dict, info: dict, origin: np.ndarray, mode: str) -> dict:
    return env_processor(create_transition(observation=_policy_observation(obs, origin, mode), info=info))


def _select_action(bundle: LoadedPolicy, transition: dict) -> dict[str, float]:
    observation = {
        key: value
        for key, value in transition[TransitionKey.OBSERVATION].items()
        if key in bundle.policy.config.input_features
    }
    observation = bundle.preprocess(observation)
    with torch.no_grad():
        action = bundle.policy.select_action(observation)
    action = bundle.postprocess(action)
    return make_robot_action(action, bundle.metadata.features)


def _relative_action_to_absolute(action: dict[str, float], origin: np.ndarray) -> dict[str, float]:
    return {
        "x.pos": float(origin[0] + action.get("x.rel", action.get("x.pos", 0.0))),
        "y.pos": float(origin[1] + action.get("y.rel", action.get("y.pos", 0.0))),
        "z.pos": float(origin[2] + action.get("z.rel", action.get("z.pos", 0.0))),
        "roll.pos": float(origin[3] + action.get("roll.rel", action.get("roll.pos", 0.0))),
        "pitch.pos": float(origin[4] + action.get("pitch.rel", action.get("pitch.pos", 0.0))),
        "yaw.pos": float(origin[5] + action.get("yaw.rel", action.get("yaw.pos", 0.0))),
        "gripper.pos": float(action.get("gripper.pos", origin[-1])),
    }


def _apply_absolute_action(env, action: dict[str, float]) -> None:
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
    xyz = np.clip(
        np.array([completed["x.pos"], completed["y.pos"], completed["z.pos"]], dtype=np.float32),
        env.ee_min,
        env.ee_max,
    )
    completed["x.pos"], completed["y.pos"], completed["z.pos"] = map(float, xyz)
    if env.use_yaw:
        yaw_offset = np.clip(completed["yaw.pos"] - env.config.fixed_yaw, env.yaw_min, env.yaw_max)
        completed["yaw.pos"] = float(env.config.fixed_yaw + yaw_offset)
    env.robot.send_action(completed)
    env.target_xyz = xyz
    env.target_yaw = (
        float(completed["yaw.pos"] - env.config.fixed_yaw)
        if env.use_yaw
        else float(completed["yaw.pos"])
    )


def _reset_to_stage(env, stage: PolicyStage, pose: list[float]) -> tuple[dict, dict, np.ndarray]:
    env.config.home_tcp = pose.copy()
    env.config.fixed_roll = float(pose[3])
    env.config.fixed_pitch = float(pose[4])
    env.config.fixed_yaw = float(pose[5])
    obs, info = env.reset()
    origin = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
    logger.info("Reset complete for %s: %s", stage.name, pose)
    return obs, info, origin


def _announce(text: str, use_tts: bool) -> None:
    print(f"[PIPELINE] {text}", flush=True)
    if use_tts:
        log_say(text)


def main() -> None:
    document = _read_json(POLICIES_CONFIG_PATH)
    pipeline = _load_pipeline_config(document)
    stages = _load_stages(document, pipeline.order)
    start_poses = _load_start_poses(stages)

    with ROBOT_CONFIG_PATH.open(encoding="utf-8") as file:
        raw_robot_config = json.load(file)
    config = draccus.decode(GymManipulatorConfig, raw_robot_config)
    if config.env.fps != pipeline.fps:
        raise ValueError(f"Robot config fps={config.env.fps} does not match pipeline fps={pipeline.fps}")

    device_name = pipeline.device if torch.cuda.is_available() or pipeline.device == "cpu" else "cpu"
    device = torch.device(device_name)
    bundles = [_load_policy(stage, pipeline, device) for stage in stages]

    env = None
    teleop_device = None
    try:
        env, teleop_device = make_robot_env(config.env)
        if teleop_device is None:
            raise RuntimeError("The robot config did not create a gamepad teleoperator")
        env.config.reset_time_s = pipeline.reset_time_s
        env_processor = _make_env_processor(config, str(device))
        dt_s = 1.0 / float(pipeline.fps)

        active_index = 0
        active = bundles[active_index]
        obs, info, origin = _reset_to_stage(env, active.stage, start_poses[active.stage.name])
        env_processor.reset()
        transition = _process_observation(
            env_processor, obs, info, origin, active.stage.action_recording_mode
        )
        _announce(f"Policy {active.stage.name}", pipeline.use_tts)
        logger.info("SUCCESS switches policy; FAILURE/RERECORD stops the pipeline.")

        success_was_pressed = False
        for step_index in range(pipeline.max_steps):
            step_start = time.perf_counter()
            events = teleop_device.get_teleop_events()
            success_pressed = bool(events.get(TeleopEvents.SUCCESS, False))

            if success_pressed and not success_was_pressed:
                active_index += 1
                if active_index >= len(bundles):
                    _announce("Complete", pipeline.use_tts)
                    break
                active = bundles[active_index]
                active.policy.reset()
                obs, info, origin = _reset_to_stage(env, active.stage, start_poses[active.stage.name])
                env_processor.reset()
                transition = _process_observation(
                    env_processor, obs, info, origin, active.stage.action_recording_mode
                )
                _announce(f"Policy {active.stage.name}", pipeline.use_tts)
                # Read the current state after the blocking reset to debounce the button.
                success_was_pressed = bool(
                    teleop_device.get_teleop_events().get(TeleopEvents.SUCCESS, False)
                )
                precise_sleep(max(dt_s - (time.perf_counter() - step_start), 0.0))
                continue

            success_was_pressed = success_pressed
            if events.get(TeleopEvents.TERMINATE_EPISODE, False):
                _announce("Stop", pipeline.use_tts)
                break

            robot_action = _select_action(active, transition)
            if active.stage.action_recording_mode == "relative_to_start":
                robot_action = _relative_action_to_absolute(robot_action, origin)
            _apply_absolute_action(env, robot_action)

            obs = env._get_observation()
            transition = _process_observation(
                env_processor,
                obs,
                {TeleopEvents.IS_INTERVENTION: False},
                origin,
                active.stage.action_recording_mode,
            )
            if step_index % pipeline.fps == 0:
                logger.info("step=%d policy=%s", step_index, active.stage.name)
            precise_sleep(max(dt_s - (time.perf_counter() - step_start), 0.0))
    except KeyboardInterrupt:
        logger.info("Stopped by Ctrl+C.")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logger.exception("teleoperator disconnect failed")


if __name__ == "__main__":
    main()
