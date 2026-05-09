"""Unit tests for the sim_assembling gym wrapper + registration."""

from __future__ import annotations

import os

import numpy as np
import pytest


def _skip_if_no_gl():
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture(scope="module")
def sim_env():
    _skip_if_no_gl()
    try:
        import gymnasium as gym

        import lerobot.envs.sim_assembling  # noqa: F401
    except ImportError:
        pytest.skip("simulator_for_il_rl / gymnasium not installed")
    try:
        env = gym.make(
            "sim_assembling/AssembleBase-v0",
            control_hz=20.0,
            mode="fast",
            max_episode_steps=20,
            render_mode="rgb_array",
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"sim env unavailable: {e}")
    yield env
    env.close()


def test_gym_registration_resolves():
    import gymnasium as gym

    import lerobot.envs.sim_assembling  # noqa: F401

    specs = [s for s in gym.envs.registry.keys() if "sim_assembling" in s]
    assert "sim_assembling/AssembleBase-v0" in specs


def test_reset_emits_gym_hil_style_obs(sim_env):
    obs, info = sim_env.reset()
    assert set(obs.keys()) == {"pixels", "agent_pos"}
    assert set(obs["pixels"].keys()) == {"front", "wrist"}
    front = obs["pixels"]["front"]
    assert front.shape == (128, 128, 3)
    assert front.dtype == np.uint8
    assert obs["agent_pos"].shape == (15,)
    assert obs["agent_pos"].dtype == np.float32


def test_action_space_is_3d_delta_plus_discrete_gripper(sim_env):
    import gymnasium as gym

    assert isinstance(sim_env.action_space, gym.spaces.Box)
    assert sim_env.action_space.shape == (4,)
    # first three are [-1, 1] deltas, last is discrete gripper [0, N-1]
    assert sim_env.action_space.low[0] == -1.0
    assert sim_env.action_space.high[0] == 1.0
    assert sim_env.action_space.low[3] == 0.0


def test_step_returns_stable_obs(sim_env):
    sim_env.reset()
    a = np.array([0.2, 0.0, -0.2, 1.0], dtype=np.float32)
    obs, r, term, trunc, info = sim_env.step(a)
    assert obs["pixels"]["front"].shape == (128, 128, 3)
    assert obs["agent_pos"].shape == (15,)
    assert isinstance(r, float)
    assert term is False
    assert trunc is False


def test_zero_action_is_passthrough_and_noncrashing(sim_env):
    sim_env.reset()
    zero = np.zeros(sim_env.action_space.shape[0], dtype=np.float32)
    for _ in range(3):
        obs, *_ = sim_env.step(zero)
    assert np.isfinite(obs["agent_pos"]).all()


def test_fast_mode_faster_than_realtime_equivalent(sim_env):
    """Fast mode must beat (by a large margin) the real-time wall-clock that
    the same control_hz would imply for the same number of steps."""
    import time

    sim_env.reset()
    n = 20
    act = np.zeros(sim_env.action_space.shape[0], dtype=np.float32)
    t0 = time.time()
    for _ in range(n):
        sim_env.step(act)
    dt = time.time() - t0
    # real-time-equivalent would be n / control_hz (20 / 20 = 1s)
    realtime_eq = n / 20.0
    assert dt < realtime_eq, f"fast mode ({dt:.3f}s) not faster than real-time ({realtime_eq:.3f}s)"


def test_ref_pose_accumulates_on_delta(sim_env):
    from lerobot.envs.sim_assembling import AssemblingHILAdapter

    # Walk the wrapper chain until we hit the adapter.
    adapter = sim_env
    while not isinstance(adapter, AssemblingHILAdapter) and hasattr(adapter, "env"):
        adapter = adapter.env
    assert isinstance(adapter, AssemblingHILAdapter)

    sim_env.reset()
    sim_env.step(np.zeros(4, dtype=np.float32))
    ref_pos0 = np.asarray(adapter._ee_ref_pos, dtype=np.float32).copy()
    sim_env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    ref_pos1 = np.asarray(adapter._ee_ref_pos, dtype=np.float32).copy()
    assert ref_pos1[0] > ref_pos0[0]
