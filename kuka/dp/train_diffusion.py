"""Train LeRobot Diffusion Policy on a KUKA LeRobot dataset."""

from pathlib import Path
import threading

import multiprocess.resource_tracker
import torch

# multiprocess 0.70.18 expects RLock._recursion_count(), which is absent in
# CPython 3.12.0 used by the local LeRobot environment.
if not hasattr(threading.RLock(), "_recursion_count"):
    def _skip_incompatible_resource_tracker_finalizer(_resource_tracker: object) -> None:
        pass

    multiprocess.resource_tracker.ResourceTracker.__del__ = _skip_incompatible_resource_tracker_finalizer

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION


DATASET_ID = "local/kuka_device_assemble2_abs_stage1"
OUTPUT_DIR = Path("outputs/device_assemble2/diffusion_abs_stage1")

# Set a checkpoint directory to continue training.
PRETRAINED_PATH: str | None = None
USE_FINETUNE_DATASET_STATS = False

DEVICE = "cuda"
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAINING_STEPS = 70_000
LOG_FREQ = 100
SAVE_FREQ = 5_000

# Temporal structure: 2 observations -> predict 16 actions -> execute 8.
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8

# DDIM gives much faster robot inference than the default 100-step DDPM.
NOISE_SCHEDULER = "DDIM"
NUM_TRAIN_TIMESTEPS = 100
NUM_INFERENCE_STEPS = 20


def make_delta_timestamps(indices: list[int] | None, fps: int) -> list[float]:
    if indices is None:
        return [0.0]
    return [index / fps for index in indices]


def save_checkpoint(policy, preprocessor, postprocessor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")
    metadata = LeRobotDatasetMetadata(DATASET_ID)
    policy_features = dataset_to_policy_features(metadata.features)
    output_features = {
        key: feature for key, feature in policy_features.items() if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in policy_features.items() if feature.type is not FeatureType.ACTION
    }

    if PRETRAINED_PATH is None:
        config = DiffusionConfig(
            input_features=input_features,
            output_features=output_features,
            device=str(device),
            n_obs_steps=N_OBS_STEPS,
            horizon=HORIZON,
            n_action_steps=N_ACTION_STEPS,
            noise_scheduler_type=NOISE_SCHEDULER,
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )
        policy = DiffusionPolicy(config)
        preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=metadata.stats)
        print("Training Diffusion Policy from scratch.")
    else:
        policy = DiffusionPolicy.from_pretrained(PRETRAINED_PATH)
        config = policy.config
        config.device = str(device)
        if USE_FINETUNE_DATASET_STATS:
            preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=metadata.stats)
        else:
            preprocessor, postprocessor = make_pre_post_processors(
                config,
                pretrained_path=PRETRAINED_PATH,
                dataset_stats=metadata.stats,
            )
        print(f"Continuing training from {PRETRAINED_PATH}.")

    policy.to(device)
    policy.train()

    observation_timestamps = make_delta_timestamps(config.observation_delta_indices, metadata.fps)
    delta_timestamps = {
        ACTION: make_delta_timestamps(config.action_delta_indices, metadata.fps),
        **{key: observation_timestamps for key in config.input_features},
    }
    dataset = LeRobotDataset(DATASET_ID, delta_timestamps=delta_timestamps)
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        drop_n_last_frames=config.drop_n_last_frames,
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    optimizer = config.get_optimizer_preset().build(policy.parameters())
    scheduler = config.get_scheduler_preset().build(optimizer, num_training_steps=TRAINING_STEPS)

    print(
        f"Dataset: {DATASET_ID} ({dataset.num_episodes} episodes, {len(sampler)} train samples)\n"
        f"Device: {device}, batch size: {BATCH_SIZE}\n"
        f"Output: {OUTPUT_DIR}"
    )

    step = 0
    while step < TRAINING_STEPS:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % LOG_FREQ == 0 or step == 1:
                learning_rate = optimizer.param_groups[0]["lr"]
                print(f"step={step} loss={loss.item():.6f} lr={learning_rate:.3e}")

            if step % SAVE_FREQ == 0:
                save_checkpoint(policy, preprocessor, postprocessor, OUTPUT_DIR / str(step))

            if step >= TRAINING_STEPS:
                break

    save_checkpoint(policy, preprocessor, postprocessor, OUTPUT_DIR / str(TRAINING_STEPS))
    save_checkpoint(policy, preprocessor, postprocessor, OUTPUT_DIR / "final")
    print("Training finished.")


if __name__ == "__main__":
    main()
