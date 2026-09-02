"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch
from torchvision.transforms import v2

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

PATH = "outputs/device_assemble2/act_abs_stage3"
DATASET_ID = "local/kuka_device_assemble2_abs_stage3"

# Set this to an already trained checkpoint directory to continue training it.
# PRETRAINED_PATH: str | None = "outputs/device_assemble2/act_stage1/70000"
PRETRAINED_PATH: str | None = None

# True: build normalization from the finetune dataset stats.
# False: reuse pre/post processors saved in PRETRAINED_PATH.
USE_FINETUNE_DATASET_STATS = False

DEVICE = "cuda"  # or "cpu"
BATCH_SIZE = 32
TRAINING_STEPS = 70000
LOG_FREQ = 100
SAVE_FREQ = 5000

USE_IMAGE_AUGMENTATIONS = True
RANDOM_SHIFT = (0.02, 0.02)  # fraction of image width/height
RANDOM_CROP_PADDING = 8  # pixels; output image size stays unchanged


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def make_image_transforms(cfg: ACTConfig):
    if not USE_IMAGE_AUGMENTATIONS or not cfg.image_features:
        return None

    image_feature = next(iter(cfg.image_features.values()))
    _, height, width = image_feature.shape

    return v2.Compose(
        [
            v2.RandomAffine(degrees=0, translate=RANDOM_SHIFT),
            v2.RandomCrop(
                size=(height, width),
                padding=RANDOM_CROP_PADDING,
                padding_mode="edge",
            ),
        ]
    )


def main():
    output_directory = Path(PATH)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device(DEVICE)

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if PRETRAINED_PATH is not None:
        print(f"Finetune from checkpoint: {PRETRAINED_PATH}")
        policy = ACTPolicy.from_pretrained(PRETRAINED_PATH)
        cfg = policy.config
        cfg.device = str(device)

        if USE_FINETUNE_DATASET_STATS:
            preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
        else:
            preprocessor, postprocessor = make_pre_post_processors(
                cfg,
                pretrained_path=PRETRAINED_PATH,
                dataset_stats=dataset_metadata.stats,
            )
    else:
        print("Train from scratch")
        cfg = ACTConfig(input_features=input_features, output_features=output_features, device=str(device))
        policy = ACTPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    image_transforms = make_image_transforms(cfg)
    if image_transforms is not None:
        print(f"Image augmentations enabled: shift={RANDOM_SHIFT}, crop_padding={RANDOM_CROP_PADDING}px")

    # Instantiate the dataset
    dataset = LeRobotDataset(
        DATASET_ID,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
    )
    # dataset = LeRobotDataset(DATASET_ID)
    print(len(dataset))

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Number of training steps and logging frequency

    # Run training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1

            if step % SAVE_FREQ == 0:
                output_directory = Path(PATH) / str(step)
                output_directory.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(output_directory)
                preprocessor.save_pretrained(output_directory)
                postprocessor.save_pretrained(output_directory)

            if step >= TRAINING_STEPS:
                done = True
                break

    # Save the policy checkpoint, alongside the pre/post processors

    output_directory = Path(PATH)
    output_directory.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    # Save all assets to the Hub
    # policy.push_to_hub("<user>/robot_learning_tutorial_act")
    # preprocessor.push_to_hub("<user>/robot_learning_tutorial_act")
    # postprocessor.push_to_hub("<user>/robot_learning_tutorial_act")


if __name__ == "__main__":
    main()
