"""Convert a HILSERL replay-buffer dataset to an ACT-compatible LeRobotDataset.

Reads one or more HILSERL output directories (``dataset/`` and/or
``dataset_offline/``), keeps only episodes where at least one frame has
``next.reward >= MIN_REWARD``, strips RL-specific fields, converts per-frame
PNG images (stored in parquet) to per-episode mp4 videos, and writes a fresh
LeRobotDataset that ``act_train/train_ur10_act_v2.py`` can consume directly.

Action conversion (delta → relative absolute target)
-----------------------------------------------------
HILSERL stores sparse bang-bang deltas: at 10 Hz the operator holds position
~90 % of the time so most deltas are near-zero.  ACT minimises L1 loss over
predicted chunks → learns the mean (≈ 0) → robot stalls during fine control.

Fix: store the *resulting target position* rather than the delta.

    state layout (17-D, use_yaw=True):
        [0:6]  joint_pos      (6) rad, absolute
        [6:12] joint_vel      (6) rad/s, absolute
        [12:15] tcp_xyz_rel   (3) m, relative to per-episode initial pose
        [15]   yaw_offset     (1) rad, relative to per-episode home yaw
        [16]   gripper        (1) 1.0=open, 0.0=closed

    new action (5-D) = next frame's measured state:
        [0] x.pos  = next_state[12]  (tcp_x_rel, metres, relative to episode start)
        [1] y.pos  = next_state[13]
        [2] z.pos  = next_state[14]
        [3] yaw.pos = next_state[15] (yaw offset, radians)
        [4] gripper.pos = next_state[16] (1.0=open, 0.0=closed)

    Why NOT state + delta: the SAC action is in [-1,1] (tanh-normalised), NOT
    metres.  Physical delta = action × 0.001 m — adding raw SAC values to the
    metre-scale tcp_xyz_rel gives a result dominated by the SAC term (≈ original
    delta).  Reading the next frame's measured position sidesteps unit conversion
    entirely and uses the ground-truth robot trajectory.

During eval, add env.robot._initial_tcp_xyz[:3] to the policy's xyz output
before calling env.set_act_target() — see eval_ur10_act_v2.py.

Usage:
    conda run -n lerobot python act_train/hilserl_to_act.py

After conversion set::

    DATASET_REPO_ID = OUTPUT_REPO_ID   # in train_ur10_act_v2.py
"""
from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


# ---------------------------------------------------------------------------
# User-tunable
# ---------------------------------------------------------------------------
HILSERL_RUN_DIR = Path("outputs/train/2026-06-01/09-18-45_pcb_hilserl_3cams_yaw")

# Each entry in SOURCE_DIRS is processed in order and appended to the output.
# Set an entry to None to skip it.
SOURCE_DIRS: list[Path | None] = [
    HILSERL_RUN_DIR / "dataset",
    HILSERL_RUN_DIR / "dataset_offline",
]

OUTPUT_REPO_ID = "local/pcb_act_3cams_yaw_from_hilserl_absact"
TASK_DESCRIPTION = "pcb_act_3cams_yaw"   # replaces "from_replay_buffer" task label
MIN_REWARD = 1.0                          # keep episodes where any frame reward >= this
FPS = 10
# ---------------------------------------------------------------------------

CAM_KEYS = [
    "observation.images.front",
    "observation.images.side",
    "observation.images.wrist",
]

# Columns loaded from each source parquet file (images last — they're the heavy ones)
_SCALAR_COLS = ["episode_index", "frame_index", "timestamp", "action", "observation.state"]
_LOAD_COLS = _SCALAR_COLS + CAM_KEYS


def _decode_png(img_struct: dict) -> np.ndarray:
    """PNG bytes stored as struct<bytes, path> → HWC uint8 numpy array."""
    return np.array(Image.open(io.BytesIO(img_struct["bytes"])))


def _next_state_to_target(next_state: np.ndarray) -> np.ndarray:
    """Extract the absolute target pose from the NEXT frame's measured state.

    Why next-state instead of (current_state + delta):
      The HILSERL replay buffer stores SAC policy outputs in [-1, 1] (normalised
      tanh space), NOT in metres.  The physical step in metres is action * 0.001,
      which is ~100× smaller than the SAC value.  Adding the raw SAC action to the
      metre-scale tcp_xyz_rel gives a result dominated by the SAC term — the
      'absolute target' would just be the original delta again.

      Instead we read the robot's MEASURED position one step later.  This is always
      in metres (same units as tcp_xyz_rel in the state), always non-zero when the
      robot has moved, and 'stay still' frames become 'target = current position'
      rather than 'target = 0'.

    state layout (17-D, use_yaw=True):
        [12:15] tcp_xyz_rel — TCP xyz relative to per-episode initial pose (metres)
        [15]    yaw_offset  — yaw offset from home (radians)
        [16]    gripper     — 1.0=open, 0.0=closed

    Returns (5,) float32: [x.pos, y.pos, z.pos, yaw.pos, gripper.pos]
    All values are relative to the per-episode start captured at env.reset().
    The eval script adds env.robot._initial_tcp_xyz[:3] to recover absolute xyz.
    """
    return np.array([
        next_state[12],   # tcp_x_rel (next measured, metres)
        next_state[13],   # tcp_y_rel
        next_state[14],   # tcp_z_rel
        next_state[15],   # yaw_offset (next measured, radians)
        next_state[16],   # gripper state (1.0=open, 0.0=closed)
    ], dtype=np.float32)


def _find_successful_episodes(source_dir: Path) -> set[int]:
    """Return episode indices that have at least one frame with reward >= MIN_REWARD."""
    data_files = sorted((source_dir / "data").glob("chunk-*/file-*.parquet"))
    successful: set[int] = set()
    for f in data_files:
        tbl = pq.read_table(f, columns=["episode_index", "next.reward"])
        ep_list = tbl["episode_index"].to_pylist()
        rw_list = tbl["next.reward"].to_pylist()
        for ep, rw in zip(ep_list, rw_list):
            if rw >= MIN_REWARD:
                successful.add(ep)
    return successful


def _collect_frames(source_dir: Path, episode_set: set[int]) -> dict[int, list[dict]]:
    """One pass over all parquet files; returns frames grouped by episode, sorted by frame_index.

    Images are kept as compressed PNG bytes to avoid materialising large tensors before
    they are needed — they are decoded lazily in the write loop below.
    Memory footprint: roughly (total_kept_frames × 3 cameras × ~20-40 KB/PNG image).
    """
    data_files = sorted((source_dir / "data").glob("chunk-*/file-*.parquet"))
    frames_by_ep: dict[int, list[dict]] = defaultdict(list)

    for f in tqdm(data_files, desc="  scanning", unit="file", leave=False):
        # Cheap pre-check: skip file if none of its episodes are in our set
        ep_col = pq.read_table(f, columns=["episode_index"])["episode_index"].to_pylist()
        if not set(ep_col).intersection(episode_set):
            continue

        tbl = pq.read_table(f, columns=_LOAD_COLS)
        ep_arr   = tbl["episode_index"].to_pylist()
        fi_arr   = tbl["frame_index"].to_pylist()
        ts_arr   = tbl["timestamp"].to_pylist()
        act_arr  = tbl["action"].to_pylist()
        st_arr   = tbl["observation.state"].to_pylist()
        img_arrs = {cam: tbl[cam].to_pylist() for cam in CAM_KEYS}
        del tbl  # release arrow buffers

        for i, ep in enumerate(ep_arr):
            if ep not in episode_set:
                continue
            frames_by_ep[ep].append({
                "frame_index":       fi_arr[i],
                "timestamp":         ts_arr[i],
                "action":            np.array(act_arr[i], dtype=np.float32),
                "observation.state": np.array(st_arr[i], dtype=np.float32),
                # keep compressed bytes; decode just before add_frame
                **{cam: img_arrs[cam][i]["bytes"] for cam in CAM_KEYS},
            })

    # Ensure frames within each episode are ordered
    for ep in frames_by_ep:
        frames_by_ep[ep].sort(key=lambda r: r["frame_index"])

    return frames_by_ep


def main() -> None:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    out_root = HF_LEROBOT_HOME / OUTPUT_REPO_ID
    if out_root.exists():
        raise FileExistsError(
            f"Output dataset already exists: {out_root}\n"
            f"Delete it first:  rm -rf {out_root}\n"
            f"Or change OUTPUT_REPO_ID at the top of this script."
        )

    features = {
        "action": {
            "dtype": "float32",
            "shape": (5,),
            "names": ["x.pos", "y.pos", "z.pos", "yaw.pos", "gripper.pos"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (17,),
            "names": None,
        },
        **{
            cam: {"dtype": "video", "shape": (3, 128, 128), "names": ["channels", "height", "width"]}
            for cam in CAM_KEYS
        },
    }

    _log(f"[convert] Creating output dataset: {OUTPUT_REPO_ID}")
    dataset = LeRobotDataset.create(OUTPUT_REPO_ID, fps=FPS, features=features, use_videos=True)
    _log(f"[convert] Output root: {dataset.root}")

    total_eps_written = 0
    total_frames_written = 0

    for source_dir in SOURCE_DIRS:
        if source_dir is None:
            continue
        source_dir = Path(source_dir)
        if not source_dir.exists():
            _log(f"[convert] WARNING: {source_dir} not found — skipping")
            continue

        _log(f"\n[convert] === {source_dir.name} ===")

        successful = _find_successful_episodes(source_dir)
        _log(f"[convert]   {len(successful)} episode(s) with reward >= {MIN_REWARD}")

        if not successful:
            continue

        _log(f"[convert]   Loading frames into memory…")
        frames_by_ep = _collect_frames(source_dir, successful)

        eps_sorted = sorted(successful)
        for ep_idx in tqdm(eps_sorted, desc=f"  {source_dir.name}", unit="ep"):
            frames = frames_by_ep.get(ep_idx)
            if not frames:
                _log(f"[convert]   WARNING: episode {ep_idx} has no frames — skipping")
                continue

            n = len(frames)
            for i, row in enumerate(frames):
                # Use NEXT frame's measured state as the target; last frame uses itself.
                next_state = frames[i + 1]["observation.state"] if i + 1 < n else row["observation.state"]
                frame = {
                    "task":              TASK_DESCRIPTION,
                    "action":            _next_state_to_target(next_state),
                    "observation.state": row["observation.state"],
                    **{cam: _decode_png({"bytes": row[cam]}) for cam in CAM_KEYS},
                }
                dataset.add_frame(frame)

            dataset.save_episode()
            total_eps_written += 1
            total_frames_written += len(frames)

        del frames_by_ep  # free memory before next source dir

    _log(f"\n[convert] Finished.")
    _log(f"[convert]   Episodes written : {total_eps_written}")
    _log(f"[convert]   Frames written   : {total_frames_written}")
    _log(f"[convert]   Output           : {dataset.root}")
    _log(f'\n[convert] To train: set DATASET_REPO_ID = "{OUTPUT_REPO_ID}" in act_train/train_ur10_act.py')


if __name__ == "__main__":
    main()
