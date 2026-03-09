"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

import matplotlib.pyplot as plt
import numpy as np

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    output_directory = Path("outputs/robot_learning_tutorial/act")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda")  # or "cuda" or "cpu"

    dataset_id = "local/test1"

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    # preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # policy.train()
    # policy.to(device)

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    print(len(dataset))

    print(dataset[0].keys())
    a = np.transpose(dataset[0]['observation.images.side'], (1, 2, 0))
    plt.imshow(a)
    plt.show()

if __name__ == "__main__":
    main()
