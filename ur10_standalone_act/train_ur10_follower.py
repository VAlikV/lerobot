"""Offline ACT training for the standalone UR10 follower dataset.

Mirrors the proven RC10 pipeline (act_train/act_training_example.py) and its most
iterated sibling act_train/train_ur10_act_v2.py: minimal config, published ACT
defaults, MEAN_STD normalization, vanilla L1 on chunked actions.

Hyperparameters (rc10-proven, kept at the ACT defaults on purpose):
  - chunk_size = 100         (3.33 s horizon at 30 Hz — matches RC10 1:1)
  - n_action_steps = 100     (full open-loop chunk replay)
  - temporal_ensemble_coeff = None
  - normalization = MEAN_STD across VISUAL / STATE / ACTION (auto from dataset stats)

NOTE: do NOT copy train_ur10_act_v2.py's chunk_size=20 — that was only because that
dataset was 10 Hz with ~120-frame episodes. This dataset is 30 Hz with ~290-frame
episodes (like RC10), so the default 100 is correct; shrinking it would cut the horizon
to ~0.67 s and hurt the policy.

    python ur10_standalone_act/train_ur10_follower.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# -- user-tunable ---------------------------------------------------------------
DATASET_REPO_ID = "local/ur10_follower_act_relative"
OUTPUT_DIR = Path("outputs/act/ur10_follower/relative")
TRAINING_STEPS = 100_000
BATCH_SIZE = 32
LOG_FREQ = 100
SAVE_FREQ = 5_000
DEVICE = "cuda"
NUM_WORKERS = 4
# Fine-tuning: set to a checkpoint dir to warm-start from existing weights (else None).
PRETRAINED_PATH: str | None = None

# ACT horizon (left at the published defaults — correct for this 30 Hz dataset).
CHUNK_SIZE = 100
N_ACTION_STEPS = 100
TEMPORAL_ENSEMBLE_COEFF: float | None = None
# -------------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Force-flush to stderr so tqdm + logs interleave correctly when piped."""
    print(msg, flush=True, file=sys.stderr)


def _make_delta_timestamps(delta_indices: list[int] | None, fps: float) -> list[float]:
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    _log("[train] Reading dataset metadata...")
    meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}
    _log(f"[train] Dataset: {meta.total_episodes} episodes, {meta.total_frames} frames @ {meta.fps} Hz")
    _log(f"[train] Inputs : { {k: tuple(v.shape) for k, v in input_features.items()} }")
    _log(f"[train] Outputs: { {k: tuple(v.shape) for k, v in output_features.items()} }")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        temporal_ensemble_coeff=TEMPORAL_ENSEMBLE_COEFF,
        device=DEVICE,
    )
    _log(f"[train] ACTConfig: chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}, "
         f"temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}, "
         f"normalization_mapping={cfg.normalization_mapping}")

    if PRETRAINED_PATH is not None:
        _log(f"[train] Loading pretrained weights from {PRETRAINED_PATH} (fine-tuning mode)")
        policy = ACTPolicy.from_pretrained(PRETRAINED_PATH)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=PRETRAINED_PATH, dataset_stats=meta.stats,
        )
    else:
        _log("[train] Building ACT policy from scratch (may download ResNet weights on first run)...")
        policy = ACTPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    policy.train()
    policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    _log(f"[train] Policy on {device} — {n_params:.1f}M params")

    # Action chunking timestamps so the dataloader returns chunked targets (+ image history
    # when configured).
    delta_timestamps = {"action": _make_delta_timestamps(cfg.action_delta_indices, meta.fps)}
    delta_timestamps |= {
        k: _make_delta_timestamps(cfg.observation_delta_indices, meta.fps) for k in cfg.image_features
    }

    _log("[train] Loading dataset (videos indexed on first pass)...")
    t0 = time.perf_counter()
    dataset = LeRobotDataset(DATASET_REPO_ID, delta_timestamps=delta_timestamps)
    _log(f"[train] Dataset loaded: {len(dataset)} sample windows in {time.perf_counter() - t0:.1f}s")

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=device.type != "cpu", drop_last=True,
    )
    _log(f"[train] Starting training: {TRAINING_STEPS} steps, batch_size={BATCH_SIZE}, "
         f"num_workers={NUM_WORKERS}; checkpoints every {SAVE_FREQ} steps -> {OUTPUT_DIR}")

    def _save(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(path)
        preprocessor.save_pretrained(path)
        postprocessor.save_pretrained(path)

    pbar = tqdm(total=TRAINING_STEPS, desc="train", unit="step", dynamic_ncols=True)
    step, done = 0, False
    try:
        while not done:
            for batch in loader:
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                loss_val = float(loss.item())
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.4f}", refresh=False)
                if step % LOG_FREQ == 0:
                    pbar.write(f"[step {step:6d}]  loss={loss_val:.4f}")
                if step % SAVE_FREQ == 0:
                    ckpt = OUTPUT_DIR / f"step_{step}"
                    _save(ckpt)
                    pbar.write(f"[checkpoint] saved -> {ckpt}")
                if step >= TRAINING_STEPS:
                    done = True
                    break
    finally:
        pbar.close()

    _save(OUTPUT_DIR / "last")
    _log(f"[train] Final checkpoint saved -> {OUTPUT_DIR / 'last'}")


if __name__ == "__main__":
    main()
