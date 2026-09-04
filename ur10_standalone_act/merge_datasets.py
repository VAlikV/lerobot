"""Merge N LeRobotDatasets recorded separately into a single dataset.

Thin, repo-aware wrapper around lerobot's `merge_datasets`
(src/lerobot/datasets/dataset_tools.py -> aggregate_datasets). It concatenates
the sources IN THE GIVEN ORDER, reindexing episode/frame indices and remapping
task indices, and writes ONE new dataset on disk. Point train_ur10_follower.py's
DATASET_REPO_ID at the output to train on the combined set (e.g. original demos +
HG-DAgger corrections from dagger_ur10_follower.py).

HARD REQUIREMENT (enforced by lerobot's validate_all_metadata): every source must
share the SAME fps, robot_type, and features (exact dict match — same state dim,
action keys, image keys + shapes). Datasets recorded with the same standalone
follower config satisfy this; an old 17-D dataset will NOT merge with a new 11-D
one — it raises ValueError on the features check.

Run (lerobot conda env):
    # default sources below:
    python ur10_standalone_act/merge_datasets.py
    # or pass N sources + an output explicitly (any number of --sources):
    python ur10_standalone_act/merge_datasets.py \
        --sources local/dsA local/dsB local/dsC \
        --output  local/dsAll
"""

from __future__ import annotations

import argparse
import logging

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- defaults (used when no CLI args are given) ---------------------------------
# List ANY number of source repo IDs here; they are merged in this order.
SOURCES = [
    "local/ur10_follower_act_relative",
    "local/ur10_follower_act_relative_dagger",
]
OUTPUT = "local/ur10_follower_act_relative_merged"
# Optional explicit root for the merged dataset; None -> $HF_LEROBOT_HOME/<OUTPUT>.
OUTPUT_DIR: str | None = None
# -------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Merge N LeRobotDatasets into one")
    p.add_argument("--sources", nargs="+", default=None,
                   help="N source repo IDs, merged in order (overrides SOURCES)")
    p.add_argument("--output", type=str, default=None,
                   help="output repo ID (overrides OUTPUT)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="root dir for the merged dataset (default $HF_LEROBOT_HOME/<output>)")
    args = p.parse_args()

    sources = args.sources if args.sources is not None else SOURCES
    output = args.output if args.output is not None else OUTPUT
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIR

    if len(sources) < 2:
        raise SystemExit(f"Need >=2 source datasets to merge, got {len(sources)}: {sources}")
    if output in sources:
        raise SystemExit(f"--output {output!r} must differ from every source repo ID.")

    logger.info("Loading %d source dataset(s)...", len(sources))
    datasets = []
    total_eps = total_frames = 0
    for repo_id in sources:
        ds = LeRobotDataset(repo_id)
        m = ds.meta
        logger.info("  %-45s  %4d eps  %7d frames  @ %d Hz  robot=%s",
                    repo_id, m.total_episodes, m.total_frames, m.fps, m.robot_type)
        total_eps += m.total_episodes
        total_frames += m.total_frames
        datasets.append(ds)

    logger.info("Merging -> %s  (expecting %d eps / %d frames)...",
                output, total_eps, total_frames)
    # merge_datasets calls validate_all_metadata first: a clear ValueError is raised
    # here if fps / robot_type / features differ across the sources.
    merged = merge_datasets(datasets, output_repo_id=output, output_dir=output_dir)

    logger.info("DONE -> %s", merged.root)
    logger.info("  merged: %d episodes, %d frames @ %d Hz",
                merged.meta.total_episodes, merged.meta.total_frames, merged.meta.fps)
    if (merged.meta.total_episodes, merged.meta.total_frames) != (total_eps, total_frames):
        logger.warning("  count mismatch vs sources (%d eps / %d frames) — inspect before training.",
                       total_eps, total_frames)


if __name__ == "__main__":
    main()
