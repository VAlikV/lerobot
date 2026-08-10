from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CONFIG_PATH = "kuka/configs/kuka_device_assemble.json"
DEVICE = "cuda"
FPS = 30
MAX_STEPS = 30000
N_ACTION_STEPS: int | None = 30
RESET_TIME_S = 2.0
USE_TTS = True


@dataclass(frozen=True)
class PolicyStage:
    name: str
    model_dir: str
    dataset_repo_id: str


POLICY_STAGES = [
    PolicyStage(
        name="stage1",
        model_dir="outputs/device_assemble2/act_abs_stage1/70000",
        dataset_repo_id="local/kuka_device_assemble2_abs_stage1",
    ),
    PolicyStage(
        name="stage2",
        model_dir="outputs/device_assemble2/act_abs_stage2/70000",
        dataset_repo_id="local/kuka_device_assemble2_abs_stage2",
    ),
    PolicyStage(
        name="stage3",
        model_dir="outputs/device_assemble2/act_abs_stage3/70000",
        dataset_repo_id="local/kuka_device_assemble2_abs_stage3",
    ),
]


def _make_env_processor(cfg: GymManipulatorConfig, device: str):
    steps = [
        Numpy2TorchActionProcessorStep(),
        VanillaObservationProcessorStep(),
    ]

    image_cfg = cfg.env.processor.image_preprocessing
    if image_cfg is not None:
        steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=image_cfg.crop_params_dict,
                resize_size=image_cfg.resize_size,
            )
        )

    steps.append(AddBatchDimensionProcessorStep())
    steps.append(DeviceProcessorStep(device=device))

    return DataProcessorPipeline[EnvTransition, EnvTransition](
        steps=steps,
        to_transition=identity_transition,
        to_output=identity_transition,
    )


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


def _processed_observation(env_processor, obs, info: dict) -> dict:
    return env_processor(create_transition(observation=obs, info=info))


def _load_policy_stage(stage: PolicyStage, device: torch.device):
    metadata = LeRobotDatasetMetadata(stage.dataset_repo_id)
    policy = ACTPolicy.from_pretrained(stage.model_dir)
    if N_ACTION_STEPS is not None:
        if N_ACTION_STEPS > policy.config.chunk_size:
            raise ValueError(
                f"{stage.name}: N_ACTION_STEPS={N_ACTION_STEPS} exceeds "
                f"chunk_size={policy.config.chunk_size}"
            )
        policy.config.n_action_steps = N_ACTION_STEPS
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=stage.model_dir,
        dataset_stats=metadata.stats,
    )
    policy.reset()
    logger.info(
        "Loaded %s from %s: chunk_size=%s n_action_steps=%s",
        stage.name,
        stage.model_dir,
        policy.config.chunk_size,
        policy.config.n_action_steps,
    )
    return metadata, policy, preprocess, postprocess


def _select_action(stage_bundle, transition: dict):
    metadata, policy, preprocess, postprocess = stage_bundle
    observation = {
        key: value
        for key, value in transition[TransitionKey.OBSERVATION].items()
        if key in policy.config.input_features
    }
    observation = preprocess(observation)
    with torch.no_grad():
        action = policy.select_action(observation)
    action = postprocess(action)
    return make_robot_action(action, metadata.features)


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    if cfg.env.fps != FPS:
        raise ValueError(f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})")

    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")
    stage_bundles = [_load_policy_stage(stage, device) for stage in POLICY_STAGES]

    env, teleop_device = make_robot_env(cfg.env)
    env.config.reset_time_s = RESET_TIME_S
    env_processor = _make_env_processor(cfg, str(device))
    dt_s = 1.0 / float(FPS)

    active_stage_idx = 0
    obs, info = env.reset()
    env_processor.reset()
    transition = _processed_observation(env_processor, obs, info)

    logger.info(
        "Running %d policies. SUCCESS button switches stage; FAILURE button stops.",
        len(POLICY_STAGES),
    )
    logger.info("Active policy: %s", POLICY_STAGES[active_stage_idx].name)
    if USE_TTS:
        log_say(f"Running {POLICY_STAGES[active_stage_idx].name}")

    try:
        for step_idx in range(MAX_STEPS):
            step_start = time.perf_counter()

            events = teleop_device.get_teleop_events() if teleop_device is not None else {}
            if events.get(TeleopEvents.SUCCESS, False):
                active_stage_idx += 1
                if active_stage_idx >= len(POLICY_STAGES):
                    logger.info("All stages completed. Stopping.")
                    if USE_TTS:
                        log_say("All stages completed")
                    break
                _metadata, policy, _preprocess, _postprocess = stage_bundles[active_stage_idx]
                policy.reset()
                obs, info = env.reset()
                env_processor.reset()
                transition = _processed_observation(env_processor, obs, info)
                logger.info("Switched to policy: %s", POLICY_STAGES[active_stage_idx].name)
                if USE_TTS:
                    log_say(f"Running {POLICY_STAGES[active_stage_idx].name}")
                precise_sleep(max(dt_s - (time.perf_counter() - step_start), 0.0))
                continue

            if events.get(TeleopEvents.TERMINATE_EPISODE, False):
                logger.info("Stop button pressed. Stopping policy execution.")
                if USE_TTS:
                    log_say("Stopping")
                break

            robot_action = _select_action(stage_bundles[active_stage_idx], transition)
            _apply_absolute_action(env, robot_action)

            obs = env._get_observation()
            transition = _processed_observation(
                env_processor,
                obs,
                {TeleopEvents.IS_INTERVENTION: False},
            )

            if step_idx % FPS == 0:
                logger.info("step=%d active_policy=%s", step_idx, POLICY_STAGES[active_stage_idx].name)

            precise_sleep(max(dt_s - (time.perf_counter() - step_start), 0.0))
    except KeyboardInterrupt:
        logger.info("Stopped by Ctrl+C.")
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logger.exception("teleop disconnect failed")


if __name__ == "__main__":
    main()
