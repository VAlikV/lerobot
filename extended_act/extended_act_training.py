"""Train Extended ACT on a dataset containing precomputed geometry."""

from __future__ import annotations

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.extended_act.configuration_act import ACTConfig, OBS_GEOMETRY
from lerobot.policies.extended_act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


OUTPUT_DIR = Path("outputs/device_assemble3/extended_act_stage1")
DATASET_ID = "local/kuka_stage_1_gray_geometry"

# Set to an Extended ACT checkpoint directory to continue training.
PRETRAINED_PATH: str | None = None

# True rebuilds normalization from the current converted dataset.
# False reuses the processors stored in PRETRAINED_PATH.
USE_FINETUNE_DATASET_STATS = False

DEVICE = "cuda"
BATCH_SIZE = 32
NUM_WORKERS = 0
TRAINING_STEPS = 70_000
LOG_FREQ = 100
SAVE_FREQ = 5_000


def select_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0.0]
    return [index / fps for index in delta_indices]


def build_input_output_features(
    dataset_metadata: LeRobotDatasetMetadata,
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: feature
        for key, feature in features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in features.items() if key not in output_features
    }

    if OBS_GEOMETRY not in input_features:
        raise KeyError(
            f"Dataset `{DATASET_ID}` does not contain `{OBS_GEOMETRY}`. "
            "Run extended_act/convert_dataset_with_geometry.py first."
        )

    geometry_shape = input_features[OBS_GEOMETRY].shape
    if len(geometry_shape) != 3 or geometry_shape[-1] != 6:
        raise ValueError(
            f"Expected geometry shape [cameras, classes, 6], got {geometry_shape}."
        )
    if OBS_GEOMETRY not in dataset_metadata.stats:
        raise KeyError(f"Dataset statistics do not contain `{OBS_GEOMETRY}`.")

    return input_features, output_features


def save_checkpoint(
    directory: Path,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)


def main() -> None:
    device = select_device(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)
    input_features, output_features = build_input_output_features(dataset_metadata)
    geometry_shape = input_features[OBS_GEOMETRY].shape

    if PRETRAINED_PATH is None:
        print("Training Extended ACT from scratch.")
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            device=str(device),
        )
        policy = ACTPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            dataset_stats=dataset_metadata.stats,
        )
    else:
        print(f"Continuing Extended ACT training from: {PRETRAINED_PATH}")
        policy = ACTPolicy.from_pretrained(PRETRAINED_PATH)
        cfg = policy.config
        cfg.device = str(device)

        if cfg.geometry_feature is None:
            raise ValueError(
                "The checkpoint does not declare observation.geometry. "
                "Use an Extended ACT checkpoint or train from scratch."
            )
        if cfg.geometry_feature.shape != geometry_shape:
            raise ValueError(
                f"Checkpoint expects geometry shape {cfg.geometry_feature.shape}, "
                f"but the dataset contains {geometry_shape}."
            )

        if USE_FINETUNE_DATASET_STATS:
            preprocessor, postprocessor = make_pre_post_processors(
                cfg,
                dataset_stats=dataset_metadata.stats,
            )
        else:
            preprocessor, postprocessor = make_pre_post_processors(
                cfg,
                pretrained_path=PRETRAINED_PATH,
                dataset_stats=dataset_metadata.stats,
            )

    policy.to(device)
    policy.train()

    delta_timestamps = {
        "action": make_delta_timestamps(
            cfg.action_delta_indices,
            dataset_metadata.fps,
        ),
        **{
            key: make_delta_timestamps(
                cfg.observation_delta_indices,
                dataset_metadata.fps,
            )
            for key in cfg.image_features
        },
    }

    # Spatial image augmentations are intentionally disabled. Geometry was
    # extracted from the original pixels, so shifting/cropping RGB alone would
    # make its centers and axes inconsistent with the images.
    dataset = LeRobotDataset(
        DATASET_ID,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
    )
    if len(dataset) == 0:
        raise RuntimeError("The training dataset is empty.")

    print(f"Dataset frames: {len(dataset)}")
    print(f"Geometry shape per frame: {geometry_shape}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    optimizer_config = cfg.get_optimizer_preset()
    optimizer = optimizer_config.build(policy.get_optim_params())
    optimizer.zero_grad(set_to_none=True)

    step = 0
    while step < TRAINING_STEPS:
        for batch in dataloader:
            if OBS_GEOMETRY not in batch:
                raise KeyError(f"Batch does not contain `{OBS_GEOMETRY}`.")

            batch = preprocessor(batch)
            loss, loss_dict = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: {loss.detach().cpu().item()}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                optimizer_config.grad_clip_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or step % LOG_FREQ == 0:
                metrics = " ".join(
                    f"{name}={value:.4f}" for name, value in loss_dict.items()
                )
                print(f"step={step} loss={loss.detach().cpu().item():.4f} {metrics}")

            if step % SAVE_FREQ == 0:
                checkpoint_dir = OUTPUT_DIR / f"checkpoint_{step:07d}"
                save_checkpoint(checkpoint_dir, policy, preprocessor, postprocessor)
                print(f"Saved checkpoint: {checkpoint_dir}")

            if step >= TRAINING_STEPS:
                break

    save_checkpoint(OUTPUT_DIR, policy, preprocessor, postprocessor)
    print(f"Training finished. Final checkpoint: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
