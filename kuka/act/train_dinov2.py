"""Train frozen DINOv2-S + ACT on a complete local kuka_put_cap dataset."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features

WORKSPACE = Path(__file__).resolve().parents[3]
CAMERAS = ("observation.images.front", "observation.images.gripper")


def positive_int(value):
    """Parse a nonzero training count."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def parse_args(argv=None):
    """Parse the standalone training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE / "kuka_put_cap")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "outputs/kuka_put_cap/act_dinov2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--steps", type=positive_int, default=70000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def validate_videos(metadata):
    """Check every referenced file, not just the first episode of each camera."""
    for key in CAMERAS:
        if key not in metadata.features or tuple(metadata.features[key]["shape"]) != (224, 224, 3):
            raise ValueError(f"Expected {key} with RGB shape (224, 224, 3)")
    paths = {
        metadata.root / metadata.get_video_file_path(ep, key)
        for ep in range(metadata.total_episodes)
        for key in CAMERAS
    }
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Missing camera videos; restore the dataset videos before training:\n" + "\n".join(missing)
        )


def main(argv=None):
    """Validate local data, train a new policy, and save its processors."""
    args = parse_args(argv)
    if not (args.dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(f"Local dataset metadata not found: {args.dataset_root / 'meta/info.json'}")
    metadata = LeRobotDatasetMetadata("local/kuka_put_cap", root=args.dataset_root)
    validate_videos(metadata)
    if metadata.fps != 30:
        raise ValueError("This experiment expects the 30 Hz kuka_put_cap dataset")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu for an offline smoke test")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    features = dataset_to_policy_features(metadata.features)
    inputs = {key: features[key] for key in ("observation.state", *CAMERAS)}
    outputs = {key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION}
    cfg = ACTConfig(
        input_features=inputs,
        output_features=outputs,
        device=args.device,
        vision_backbone="dinov2",
        use_relative_actions=False,
        chunk_size=60,
        n_action_steps=1,
        temporal_ensemble_coeff=None,
        dim_model=512,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        kl_weight=10,
        dropout=0.1,
        optimizer_lr=1e-5,
        optimizer_weight_decay=1e-4,
    )
    dataset = LeRobotDataset(
        "local/kuka_put_cap",
        root=args.dataset_root,
        delta_timestamps={"action": [i / metadata.fps for i in cfg.action_delta_indices]},
    )
    if len(dataset) < args.batch_size:
        raise ValueError("Dataset has fewer frames than batch size")
    policy = ACTPolicy(cfg).to(args.device).train()
    pre, post = make_pre_post_processors(cfg, dataset_stats=metadata.stats)
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    trainable = [p for p in policy.parameters() if p.requires_grad]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=args.device.startswith("cuda"),
    )

    def save(path):
        path.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(path)
        pre.save_pretrained(path)
        post.save_pretrained(path)

    step = 0
    while step < args.steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = policy(pre(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(
                    f"step={step} loss={loss.item():.6f} "
                    f"l1={metrics['l1_loss']:.6f} kl={metrics['kld_loss']:.6f}",
                    flush=True,
                )
            if step % 5000 == 0:
                save(args.output_dir / str(step))
            if step >= args.steps:
                break
    save(args.output_dir)


if __name__ == "__main__":
    main()
