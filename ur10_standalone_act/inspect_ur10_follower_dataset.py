"""Comprehensive inspection of a UR10-follower relative-ACT dataset.

Validates the things that actually break ACT before you waste a training run:
  - schema (features, shapes, names, dtypes) and basic counts
  - per-dim action stats + how much of the action signal is near-zero
    ("bang-bang" check — the failure that motivated relative actions)
  - RELATIVE-action sanity: every episode's FIRST frame should have xyz ~ 0
    (home is captured at the start of each record window)
  - observation.state per-dim ranges
  - image keys/shapes/dtype/value-range, plus a saved montage (front/side/wrist
    across several frames) so you can eyeball the hardcoded crop boxes
  - optional Rerun replay of decoded frames (cropped 224x224 images + signals)

Usage:
    python ur10_standalone_act/inspect_ur10_follower_dataset.py \
        --repo_id local/ur10_follower_act_relative [--montage_frames 6] [--rerun]
"""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _to_hwc_uint8(img) -> np.ndarray:
    """Dataset image (tensor/ndarray, CHW or HWC, float[0,1] or uint8) -> HWC uint8 RGB."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 + 1e-6 else 1.0), 0, 255).astype(np.uint8)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a UR10-follower relative-ACT dataset")
    ap.add_argument("--repo_id", type=str, default="local/ur10_follower_act_relative")
    ap.add_argument("--montage_frames", type=int, default=6)
    ap.add_argument("--zero_eps", type=float, default=1e-4,
                    help="|action| below this counts as 'near-zero' (bang-bang probe)")
    ap.add_argument("--rerun", action="store_true", help="replay decoded frames to Rerun")
    ap.add_argument("--rerun_max", type=int, default=300, help="max frames to replay to Rerun")
    args = ap.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id)
    print("=" * 70)
    print(f"DATASET: {args.repo_id}")
    print("=" * 70)
    print(f"  robot_type : {meta.robot_type}")
    print(f"  fps        : {meta.fps}")
    print(f"  episodes   : {meta.total_episodes}")
    print(f"  frames     : {meta.total_frames}")
    if meta.total_episodes:
        print(f"  avg ep len : {meta.total_frames / meta.total_episodes:.1f} frames "
              f"({meta.total_frames / meta.total_episodes / meta.fps:.1f}s)")

    print("\n-- FEATURES --")
    for key, ft in meta.features.items():
        names = ft.get("names")
        ncol = f" names={names}" if names and len(names) <= 12 else (f" ({len(names)} names)" if names else "")
        print(f"  {key:28s} {ft['dtype']:8s} {tuple(ft['shape'])}{ncol}")

    # ---- stats from metadata ----
    def _print_stats(key):
        s = meta.stats.get(key)
        if not s:
            return
        names = meta.features.get(key, {}).get("names") or [f"{i}" for i in range(len(np.atleast_1d(s["mean"])))]
        print(f"\n-- {key} per-dim stats (from metadata) --")
        mean, std = np.atleast_1d(s["mean"]), np.atleast_1d(s["std"])
        lo, hi = np.atleast_1d(s["min"]), np.atleast_1d(s["max"])
        for i, nm in enumerate(names):
            print(f"  {nm:14s} mean={mean[i]:+.4f} std={std[i]:.4f} min={lo[i]:+.4f} max={hi[i]:+.4f}")

    _print_stats("action")
    _print_stats("observation.state")

    # ---- frame-level analysis from hf_dataset (no video decode) ----
    ds = LeRobotDataset(args.repo_id)
    hf = ds.hf_dataset
    actions = np.asarray(hf["action"], dtype=np.float32)          # (N, 5)
    ep_idx = np.asarray(hf["episode_index"], dtype=np.int64)       # (N,)
    act_names = meta.features["action"]["names"]

    print("\n-- ACTION near-zero fraction (bang-bang probe; relative should be mostly NON-zero) --")
    for i, nm in enumerate(act_names):
        frac = float(np.mean(np.abs(actions[:, i]) < args.zero_eps))
        print(f"  {nm:14s} |a|<{args.zero_eps:g}: {frac*100:5.1f}%   "
              f"range [{actions[:, i].min():+.4f}, {actions[:, i].max():+.4f}]")

    print("\n-- RELATIVE-action sanity: first frame of each episode should have xyz ~ 0 --")
    first_norms = []
    for e in np.unique(ep_idx):
        first = actions[np.argmax(ep_idx == e)]
        first_norms.append(float(np.linalg.norm(first[:3])))
    first_norms = np.array(first_norms)
    print(f"  episodes: {len(first_norms)}  first-frame |xyz|: "
          f"mean={first_norms.mean()*1000:.2f}mm max={first_norms.max()*1000:.2f}mm")
    bad = np.where(first_norms > 0.01)[0]
    if len(bad):
        print(f"  WARNING: {len(bad)} episode(s) start >10mm from home (home capture issue?): "
              f"{bad.tolist()[:10]}")
    else:
        print("  OK: all episodes start ~at home.")

    # episode length distribution
    lengths = np.array([int(np.sum(ep_idx == e)) for e in np.unique(ep_idx)])
    print(f"\n-- EPISODE LENGTHS -- min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.1f} (frames)")

    # ---- image check + montage (decodes video) ----
    img_keys = [k for k in meta.features if "image" in k]
    print(f"\n-- IMAGES -- keys: {img_keys}")
    n = len(ds)
    sample_idx = np.linspace(0, n - 1, min(args.montage_frames, n)).astype(int) if n else []
    rows = []
    for si, idx in enumerate(sample_idx):
        frame = ds[int(idx)]
        cols = []
        for k in img_keys:
            im = _to_hwc_uint8(frame[k])
            if si == 0:
                print(f"  {k:28s} shape={im.shape} dtype={im.dtype} range=[{im.min()},{im.max()}]")
            cols.append(im)
        if cols:
            rows.append(np.concatenate(cols, axis=1))
    if rows:
        try:
            import cv2
            montage = np.concatenate(rows, axis=0)
            out = "ur10_standalone_act/dataset_inspect_montage.png"
            cv2.imwrite(out, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            print(f"  montage ({len(rows)} frames x {len(img_keys)} cams) saved -> {out}")
        except Exception as e:
            print(f"  montage save skipped: {e}")

    # ---- optional Rerun replay ----
    if args.rerun:
        from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
        init_rerun(session_name=f"inspect_{args.repo_id.replace('/', '_')}")
        state_names = meta.features["observation.state"]["names"]
        for idx in range(min(n, args.rerun_max)):
            frame = ds[int(idx)]
            obs = {k: _to_hwc_uint8(frame[k]) for k in img_keys}
            st = np.asarray(frame["observation.state"])
            obs |= {state_names[i]: float(st[i]) for i in range(len(state_names))}
            act = np.asarray(frame["action"])
            action = {act_names[i]: float(act[i]) for i in range(len(act_names))}
            log_rerun_data(observation=obs, action=action, compress_images=False)
        print(f"  replayed {min(n, args.rerun_max)} frames to Rerun.")

    print("\nDone.")


if __name__ == "__main__":
    main()
