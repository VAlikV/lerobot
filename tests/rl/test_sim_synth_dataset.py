"""Verify synthetic dataset generator produces usable LeRobotDatasets for
binary-classifier and SARM-data-prep flows."""

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


def test_generate_success_dataset(ensure_sim, tmp_path_factory):
    from tests.fixtures.sim_synth_dataset import generate_sim_synth_dataset

    root = tmp_path_factory.mktemp("synth_success")
    ds = generate_sim_synth_dataset(
        root=root, repo_id="local/sim_success", kind="success", num_episodes=2, ep_len=10
    )
    assert ds.num_episodes == 2
    # reward channel should be 0 everywhere except the final frame of each episode.
    rewards = [float(ds[i]["next.reward"].item()) for i in range(len(ds))]
    assert sum(r > 0.5 for r in rewards) == 2  # one success per ep
    dones = [bool(ds[i]["next.done"].item()) for i in range(len(ds))]
    assert sum(dones) == 2


def test_generate_failure_dataset(ensure_sim, tmp_path_factory):
    from tests.fixtures.sim_synth_dataset import generate_sim_synth_dataset

    root = tmp_path_factory.mktemp("synth_failure")
    ds = generate_sim_synth_dataset(
        root=root, repo_id="local/sim_failure", kind="failure", num_episodes=2, ep_len=10
    )
    assert ds.num_episodes == 2
    rewards = [float(ds[i]["next.reward"].item()) for i in range(len(ds))]
    assert all(r == 0.0 for r in rewards)
    dones = [bool(ds[i]["next.done"].item()) for i in range(len(ds))]
    assert not any(dones)
