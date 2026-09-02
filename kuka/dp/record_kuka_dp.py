"""Run Diffusion Policy on KUKA and record gamepad interventions (DAgger)."""

import logging

import torch

from kuka.act import record_kuka_act3 as recorder
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


# KUKA, cameras, workspace limits and gamepad configuration.
CONFIG_PATH = "kuka/dp/configs/kuka_device_assemble_dp.json"

# Trained Diffusion checkpoint and the dataset whose statistics/action names it uses.
POLICY_DIR = "outputs/device_assemble2/diffusion_abs_stage1/final"
POLICY_DATASET_REPO_ID = "local/kuka_device_assemble2_abs_stage1"
POLICY_DEVICE = "cuda"

# Dataset containing policy rollouts plus human corrections.
REPO_ID = "local/kuka_device_assemble_diffusion_dagger"
TASK_DESCRIPTION = "kuka_assemble"
NUM_EPISODES = 20
EPISODE_TIME_S = 60
FPS = 30

# Must not exceed horizon - n_obs_steps + 1 (15 for the default 2/16 setup).
N_ACTION_STEPS = 8
NUM_INFERENCE_STEPS = 20

USE_TTS = True
SHOW_FORCE_VECTOR = True


def _load_diffusion_policy(policy_dir: str, dataset_repo_id: str, requested_device: str):
    metadata = LeRobotDatasetMetadata(dataset_repo_id)
    policy = DiffusionPolicy.from_pretrained(policy_dir)
    max_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    if not 1 <= N_ACTION_STEPS <= max_action_steps:
        raise ValueError(
            f"N_ACTION_STEPS must be in [1, {max_action_steps}], got {N_ACTION_STEPS}."
        )

    policy.config.n_action_steps = N_ACTION_STEPS
    policy.diffusion.num_inference_steps = NUM_INFERENCE_STEPS
    device = torch.device(requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu")
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_dir,
        dataset_stats=metadata.stats,
    )
    policy.reset()
    logging.info(
        "Diffusion Policy loaded from %s on %s: horizon=%d, action_steps=%d, inference_steps=%d",
        policy_dir,
        device,
        policy.config.horizon,
        policy.config.n_action_steps,
        policy.diffusion.num_inference_steps,
    )
    return metadata, policy, preprocessor, postprocessor, device


def main() -> None:
    # Reuse the hardware, safety, recording and intervention implementation of
    # record_kuka_act3; only policy loading/inference differs between ACT and DP.
    recorder.CONFIG_PATH = CONFIG_PATH
    recorder.POLICY_DIR = POLICY_DIR
    recorder.POLICY_DATASET_REPO_ID = POLICY_DATASET_REPO_ID
    recorder.POLICY_DEVICE = POLICY_DEVICE
    recorder.REPO_ID = REPO_ID
    recorder.TASK_DESCRIPTION = TASK_DESCRIPTION
    recorder.NUM_EPISODES = NUM_EPISODES
    recorder.EPISODE_TIME_S = EPISODE_TIME_S
    recorder.FPS = FPS
    recorder.N_ACTION_STEPS = N_ACTION_STEPS
    recorder.USE_TTS = USE_TTS
    recorder.SHOW_FORCE_VECTOR = SHOW_FORCE_VECTOR
    recorder._load_policy = _load_diffusion_policy
    recorder.main()


if __name__ == "__main__":
    main()
