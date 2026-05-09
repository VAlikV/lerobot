"""Unit tests for HeightGripperRewardStep."""

from __future__ import annotations

import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.reward_model import (
    HeightGripperRewardConfig,
    HeightGripperRewardStep,
)
from lerobot.utils.constants import OBS_STATE


def _make_state(z: float, gripper: float, dim: int = 8, z_idx: int = 2, g_idx: int = 7) -> torch.Tensor:
    s = torch.zeros(dim)
    s[z_idx] = z
    s[g_idx] = gripper
    return s


def _trans(state: torch.Tensor):
    return {
        TransitionKey.OBSERVATION.value: {OBS_STATE: state},
        TransitionKey.REWARD.value: 0.0,
        TransitionKey.DONE.value: False,
        TransitionKey.TRUNCATED.value: False,
        TransitionKey.INFO.value: {},
    }


def test_lifted_and_closed_gives_reward():
    cfg = HeightGripperRewardConfig(height_threshold=0.2, gripper_closed_threshold=0.5)
    step = HeightGripperRewardStep(config=cfg)
    out = step(_trans(_make_state(z=0.25, gripper=0.1)))
    assert out[TransitionKey.REWARD.value] == 1.0
    assert out[TransitionKey.DONE.value] is True


def test_lifted_but_open_gives_zero():
    cfg = HeightGripperRewardConfig(height_threshold=0.2, gripper_closed_threshold=0.5)
    step = HeightGripperRewardStep(config=cfg)
    out = step(_trans(_make_state(z=0.25, gripper=0.8)))
    assert out[TransitionKey.REWARD.value] == 0.0
    assert out[TransitionKey.DONE.value] is False


def test_low_and_closed_gives_zero():
    cfg = HeightGripperRewardConfig(height_threshold=0.2, gripper_closed_threshold=0.5)
    step = HeightGripperRewardStep(config=cfg)
    out = step(_trans(_make_state(z=0.05, gripper=0.1)))
    assert out[TransitionKey.REWARD.value] == 0.0


def test_custom_indices_are_respected():
    cfg = HeightGripperRewardConfig(
        height_threshold=0.2,
        gripper_closed_threshold=0.5,
        z_index=14,
        gripper_index=17,
    )
    step = HeightGripperRewardStep(config=cfg)
    s = torch.zeros(25)
    s[14] = 0.4
    s[17] = 0.2
    out = step(_trans(s))
    assert out[TransitionKey.REWARD.value] == 1.0


def test_batched_state_is_flattened():
    cfg = HeightGripperRewardConfig(height_threshold=0.2, gripper_closed_threshold=0.5)
    step = HeightGripperRewardStep(config=cfg)
    state = _make_state(z=0.3, gripper=0.1).unsqueeze(0)  # [1, 8]
    out = step(_trans(state))
    assert out[TransitionKey.REWARD.value] == 1.0


def test_terminate_false_still_sets_reward():
    cfg = HeightGripperRewardConfig(height_threshold=0.2, gripper_closed_threshold=0.5)
    step = HeightGripperRewardStep(config=cfg, terminate_on_success=False)
    out = step(_trans(_make_state(z=0.3, gripper=0.1)))
    assert out[TransitionKey.REWARD.value] == 1.0
    assert out[TransitionKey.DONE.value] is False
