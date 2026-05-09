"""End-to-end integration: generate synth sim data → split → verify usable.

Confirms the port utilities (split_dataset) accept a fresh sim dataset.
"""

from __future__ import annotations

import os

import pytest


def _skip_if_no_gl():
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture(scope="module")
def ensure_sim():
    _skip_if_no_gl()
    try:
        import lerobot.envs.sim_assembling  # noqa: F401
    except ImportError:
        pytest.skip("simulator_for_il_rl not installed")


def test_split_synth_dataset(ensure_sim, tmp_path_factory):
    from lerobot.policies.sac.reward_model.split_dataset import split_dataset
    from tests.fixtures.sim_synth_dataset import generate_sim_synth_dataset

    root = tmp_path_factory.mktemp("synth_for_split")
    generate_sim_synth_dataset(
        root=root, repo_id="local/sim_for_split", kind="success", num_episodes=3, ep_len=8
    )
    train_ds, val_ds = split_dataset(
        src_repo_id="local/sim_for_split",
        src_root=str(root),
        train_repo_id="local/sim_for_split-train",
        val_repo_id="local/sim_for_split-val",
        val_stride=4,
    )
    assert train_ds.num_frames > 0
    assert val_ds.num_frames > 0
    # With stride=4 and 24 frames total, expect roughly 75/25.
    total = train_ds.num_frames + val_ds.num_frames
    assert total == 24  # 3 episodes × 8 frames
    assert val_ds.num_frames / total == pytest.approx(0.25, abs=0.05)
