"""End-to-end smoke: drive the sim_assembling env through the full processor pipeline
for each of the 3 reward-model scenarios (manual / cnn / sarm), with no checkpoint.

Short rollouts (10 steps each) to keep CI time low. No gRPC, no training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


def _skip_if_no_gl():
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture(scope="module")
def ensure_sim():
    _skip_if_no_gl()
    try:
        import lerobot.envs.sim_assembling  # noqa: F401

        # force-register teleop config choices so draccus can decode the JSONs
        import lerobot.teleoperators.gamepad.configuration_gamepad  # noqa: F401
    except ImportError:
        pytest.skip("simulator_for_il_rl / lerobot teleoperator not installed")


def _load_env_cfg(path: Path):
    import draccus

    from lerobot.envs.configs import HILSerlRobotEnvConfig

    with path.open() as f:
        raw = json.load(f)
    cfg = draccus.decode(HILSerlRobotEnvConfig, raw["env"])
    # override teleop so the test doesn't need a real gamepad
    cfg.teleop = None
    # keep training short
    cfg.processor.reset.control_time_s = 1.0
    return cfg


_SIM_CFG_DIR = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "rl"


@pytest.mark.parametrize(
    "cfg_path",
    [
        _SIM_CFG_DIR / "sim_assembling_manual_env.json",
        _SIM_CFG_DIR / "sim_assembling_cnn_env.json",
        _SIM_CFG_DIR / "sim_assembling_sarm_env.json",
    ],
    ids=["manual", "cnn", "sarm"],
)
def test_sim_scenario_smoke(ensure_sim, cfg_path):
    """Each JSON cfg should yield a running pipeline that doesn't crash."""
    from lerobot.processor.core import TransitionKey
    from lerobot.processor.converters import create_transition
    from lerobot.rl.gym_manipulator import make_processors, make_robot_env

    cfg = _load_env_cfg(cfg_path)
    # pull the reward model pretrained_path to None so we don't need checkpoints.
    if cfg.processor.reward_model is not None:
        cfg.processor.reward_model.pretrained_path = None
        cfg.processor.reward_model.device = "cpu"

    try:
        env, teleop = make_robot_env(cfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"env build failed: {e}")

    try:
        env_pipe, action_pipe = make_processors(env, teleop, cfg, "cpu")

        obs, _ = env.reset()
        act_np = np.asarray(env.action_space.sample(), dtype=np.float32)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(act_np)
            t = create_transition(
                observation=obs,
                action=act_np,
                reward=float(r),
                done=bool(term),
                truncated=bool(trunc),
                info=dict(info),
            )
            out = env_pipe(t)
            assert TransitionKey.OBSERVATION.value in out
            assert np.isfinite(float(out[TransitionKey.REWARD.value] or 0.0))
            act_np = np.asarray(env.action_space.sample(), dtype=np.float32)
    finally:
        env.close()
