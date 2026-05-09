"""Sim integration for reward-model processors against gym_hil's PandaPickCubeBase-v0.

Uses the non-gamepad env to avoid SDL/X11 dependencies. Rendering uses
``MUJOCO_GL=egl`` — set the env var in the test runner.
"""

from __future__ import annotations

import os

import pytest
import torch


def _skip_if_no_gl():
    if "MUJOCO_GL" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture(scope="module")
def panda_env():
    _skip_if_no_gl()
    try:
        import gym_hil  # noqa: F401
        import gymnasium as gym
    except ImportError:
        pytest.skip("gym_hil / gymnasium not installed")
    try:
        env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True, render_mode="rgb_array")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"gym_hil env unavailable: {e}")
    yield env
    env.close()


def _build_obs_for_reward_step(raw_obs: dict) -> dict:
    """Flatten gym_hil obs into the key shape reward-model steps expect."""
    out = {}
    # Images: gym_hil nests them under "pixels". Expose them as observation.images.*
    if "pixels" in raw_obs:
        for k, v in raw_obs["pixels"].items():
            t = torch.from_numpy(v).permute(2, 0, 1).float() / 255.0  # (H, W, 3) -> (3, H, W)
            out[f"observation.images.{k}"] = t
    if "agent_pos" in raw_obs:
        out["observation.state"] = torch.from_numpy(raw_obs["agent_pos"]).float()
    return out


def test_height_gripper_runs_on_gym_hil_env(panda_env):
    """HeightGripperRewardStep runs without crashing on real sim observations."""
    from lerobot.processor.core import TransitionKey
    from lerobot.processor.reward_model import (
        HeightGripperRewardConfig,
        HeightGripperRewardStep,
    )

    raw_obs, _ = panda_env.reset()
    obs = _build_obs_for_reward_step(raw_obs)
    assert "observation.state" in obs

    # gym_hil agent_pos layout: gripper q-pos at index 18 is 2 joints (7 arm + 2 gripper)
    # 25-dim layout per fork's train_hil_serl_env.json: state.shape=[25] means
    # concatenation includes EE pose. Here PandaPickCubeBase-v0 gives 18 dims.
    # Probe to locate a sensible z/gripper pair: agent_pos[18]=0 (just right of arm).
    # For smoke, pick impossible thresholds so reward=0 every step (deterministic).
    step = HeightGripperRewardStep(
        config=HeightGripperRewardConfig(
            height_threshold=99.0,
            gripper_closed_threshold=-99.0,
            z_index=min(2, obs["observation.state"].shape[-1] - 1),
            gripper_index=min(7, obs["observation.state"].shape[-1] - 1),
        ),
        terminate_on_success=False,
    )
    t = {
        TransitionKey.OBSERVATION.value: obs,
        TransitionKey.REWARD.value: 0.0,
        TransitionKey.DONE.value: False,
        TransitionKey.TRUNCATED.value: False,
        TransitionKey.INFO.value: {},
    }
    out = step(t)
    assert out[TransitionKey.REWARD.value] == 0.0
    assert "reward_classifier_frequency" in out[TransitionKey.INFO.value]


def test_cnn_step_no_checkpoint_passes_through(panda_env):
    """CNNRewardProcessorStep with pretrained_path=None returns 0.0 reward."""
    from lerobot.processor.core import TransitionKey
    from lerobot.processor.reward_model import CNNRewardConfig, CNNRewardProcessorStep

    raw_obs, _ = panda_env.reset()
    obs = _build_obs_for_reward_step(raw_obs)
    step = CNNRewardProcessorStep(
        config=CNNRewardConfig(pretrained_path=None),
        terminate_on_success=False,
    )
    t = {
        TransitionKey.OBSERVATION.value: obs,
        TransitionKey.REWARD.value: 0.0,
        TransitionKey.DONE.value: False,
        TransitionKey.TRUNCATED.value: False,
        TransitionKey.INFO.value: {},
    }
    out = step(t)
    assert out[TransitionKey.REWARD.value] == 0.0
    assert out[TransitionKey.DONE.value] is False


def test_gym_hil_rollout_with_reward_step(panda_env):
    """5 random actions + HeightGripperRewardStep injected — no crash, reward channel populated."""
    from lerobot.processor.core import TransitionKey
    from lerobot.processor.reward_model import (
        HeightGripperRewardConfig,
        HeightGripperRewardStep,
    )

    step = HeightGripperRewardStep(
        config=HeightGripperRewardConfig(
            height_threshold=99.0,
            gripper_closed_threshold=-99.0,
            z_index=0,
            gripper_index=0,
        ),
        terminate_on_success=False,
    )
    raw_obs, _ = panda_env.reset()
    for _ in range(5):
        a = panda_env.action_space.sample()
        raw_obs, env_r, term, trunc, info = panda_env.step(a)
        obs = _build_obs_for_reward_step(raw_obs)
        t = {
            TransitionKey.OBSERVATION.value: obs,
            TransitionKey.REWARD.value: float(env_r),
            TransitionKey.DONE.value: bool(term),
            TransitionKey.TRUNCATED.value: bool(trunc),
            TransitionKey.INFO.value: dict(info),
        }
        out = step(t)
        assert TransitionKey.REWARD.value in out
        assert "reward_classifier_frequency" in out[TransitionKey.INFO.value]
