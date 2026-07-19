"""Drop dimensions from a LeRobotDataset's `observation.state` WITHOUT re-recording.

Use case: a dataset was recorded with the 17-D UR10 HIL-SERL state
    [joint_pos(6), joint_vel(6), tcp_xyz_rel(3), yaw(1), gripper(1)]
and you now want the 11-D layout (drop the 6 joint velocities, indices 6:12)
    [joint_pos(6), tcp_xyz_rel(3), yaw(1), gripper(1)]
to match a policy/env built with `include_joint_velocities=False`.

Why surgical (not modify_features / re-record):
  - `observation.state` is ONE vector feature; lerobot's modify_features can't
    overwrite a feature in place (it rejects re-adding an existing name and drops
    removed columns before its callbacks run).
  - Images are stored as separate VIDEO files referenced by the parquet, so editing
    only the `observation.state` parquet column touches NO video data — no re-encode,
    no quality loss, fast.

This edits a COPY (the source dataset is never modified). It rewrites every place the
v3.0 format stores `observation.state` numbers, sliced to the kept indices:
  1. data/*/*.parquet                              — the per-frame state column
  2. meta/episodes/**/*.parquet                    — per-episode `stats/observation.state/*`
  3. meta/stats.json                               — aggregated observation.state stats
  4. meta/info.json                                — features[...].shape (+ names if present)
Any observation.state-associated array whose length == the source dim is sliced;
scalars like `count` (length 1) are left alone.

Run on the machine that holds the dataset:
    python act_train/slim_observation_state.py \
        --src ~/.cache/huggingface/lerobot/local/<your_dataset> \
        --dst ~/.cache/huggingface/lerobot/local/<your_dataset>_11d
    # default drops indices [6:12) (the 6 joint velocities). Override with --drop 6 12.

After it finishes, point your training/JSON `repo_id` at the _11d dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

STATE_KEY = "observation.state"


def _keep_indices(src_dim: int, drop_lo: int, drop_hi: int) -> list[int]:
    return [i for i in range(src_dim) if not (drop_lo <= i < drop_hi)]


def _slice_cell(val, keep: list[int], src_dim: int):
    """Slice a single array-valued cell to `keep` iff it has length src_dim; else return as-is."""
    arr = np.asarray(val)
    if arr.ndim == 1 and arr.shape[0] == src_dim:
        return arr[keep]
    return val


def main() -> None:
    p = argparse.ArgumentParser(description="Drop dims from a LeRobotDataset observation.state")
    p.add_argument("--src", type=Path, required=True, help="source dataset root (dir with meta/ data/)")
    p.add_argument("--dst", type=Path, required=True, help="output dataset root (must not exist)")
    p.add_argument("--drop", type=int, nargs=2, default=[6, 12], metavar=("LO", "HI"),
                   help="drop observation.state indices [LO:HI). Default 6 12 = the 6 joint vels.")
    args = p.parse_args()

    src, dst = args.src, args.dst
    drop_lo, drop_hi = args.drop
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"{src} doesn't look like a LeRobotDataset (no meta/info.json)")
    if dst.exists():
        raise SystemExit(f"--dst {dst} already exists; pick a fresh path or delete it first")

    # --- read source schema to get the state dim ---
    info = json.loads((src / "meta" / "info.json").read_text())
    feat = info["features"].get(STATE_KEY)
    if feat is None:
        raise SystemExit(f"{STATE_KEY} not found in features: {list(info['features'])}")
    src_dim = int(feat["shape"][0])
    keep = _keep_indices(src_dim, drop_lo, drop_hi)
    new_dim = len(keep)
    print(f"source {STATE_KEY} dim = {src_dim}; dropping [{drop_lo}:{drop_hi}) -> keep {new_dim} "
          f"indices {keep}")
    if new_dim == src_dim:
        raise SystemExit("nothing to drop (check --drop range vs source dim)")

    # --- copy the whole dataset (videos copied as-is, original untouched) ---
    print(f"copying {src} -> {dst} ...")
    shutil.copytree(src, dst)

    # 1) data parquet: per-frame state column ---------------------------------
    data_files = sorted((dst / "data").glob("**/*.parquet"))
    if not data_files:
        raise SystemExit(f"no data parquet files under {dst/'data'}")
    n_frames = 0
    for f in data_files:
        df = pd.read_parquet(f)
        if STATE_KEY not in df.columns:
            raise SystemExit(f"{STATE_KEY} column missing in {f}")
        df[STATE_KEY] = [np.asarray(v)[keep].astype(np.float32) for v in df[STATE_KEY]]
        df.to_parquet(f, index=False)
        n_frames += len(df)
    print(f"  [1/4] rewrote {STATE_KEY} in {len(data_files)} data file(s), {n_frames} frames")

    # 2) per-episode stats parquet: stats/observation.state/* -----------------
    ep_files = sorted((dst / "meta" / "episodes").glob("**/*.parquet"))
    stat_cols_done = 0
    for f in ep_files:
        df = pd.read_parquet(f)
        cols = [c for c in df.columns if c.startswith(f"stats/{STATE_KEY}/")]
        for c in cols:
            df[c] = [_slice_cell(v, keep, src_dim) for v in df[c]]
        if cols:
            df.to_parquet(f, index=False)
            stat_cols_done += len(cols)
    print(f"  [2/4] sliced {stat_cols_done} per-episode stat column(s) across {len(ep_files)} file(s)")

    # 3) aggregated meta/stats.json ------------------------------------------
    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        s = stats.get(STATE_KEY, {})
        for k, v in list(s.items()):
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.shape[0] == src_dim:
                s[k] = arr[keep].tolist()
        stats[STATE_KEY] = s
        stats_path.write_text(json.dumps(stats))
        print(f"  [3/4] sliced meta/stats.json {STATE_KEY} -> {sorted(s)}")
    else:
        print("  [3/4] no meta/stats.json (skipped)")

    # 4) meta/info.json: shape (+ names) -------------------------------------
    feat["shape"] = [new_dim]
    names = feat.get("names")
    if isinstance(names, list) and len(names) == src_dim:
        feat["names"] = [names[i] for i in keep]
    info["features"][STATE_KEY] = feat
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    print(f"  [4/4] info.json {STATE_KEY}.shape -> [{new_dim}]")

    # --- verify a sample frame ---
    sample = pd.read_parquet(data_files[0])[STATE_KEY].iloc[0]
    print(f"\nDONE -> {dst}\n  sample {STATE_KEY} length now = {len(np.asarray(sample))} (expected {new_dim})")
    print("  Point your training repo_id at this dataset. Original left untouched.")


if __name__ == "__main__":
    main()
