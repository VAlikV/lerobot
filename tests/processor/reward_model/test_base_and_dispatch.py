"""Unit tests for BaseRewardProcessorStep + build_reward_model_step dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lerobot.processor.core import TransitionKey
from lerobot.processor.reward_model import (
    BaseRewardProcessorStep,
    CNNRewardProcessorStep,
    HeightGripperRewardStep,
    RewardModelConfig,
    SARMRewardProcessorStep,
    build_reward_model_step,
)


@dataclass
class _FakeRewardStep(BaseRewardProcessorStep):
    config: RewardModelConfig = field(default_factory=lambda: RewardModelConfig(success_reward=2.5))
    fixed_reward: float = 0.0

    def compute_reward(self, observation: dict[str, Any]) -> float:
        return self.fixed_reward


def _trans(**extra):
    base = {
        TransitionKey.OBSERVATION.value: {"observation.state": None},
        TransitionKey.REWARD.value: 0.0,
        TransitionKey.DONE.value: False,
        TransitionKey.TRUNCATED.value: False,
        TransitionKey.INFO.value: {},
    }
    base.update(extra)
    return base


def test_passthrough_when_no_observation():
    step = _FakeRewardStep(fixed_reward=1.0)
    t = _trans(**{TransitionKey.OBSERVATION.value: None})
    out = step(t)
    assert out[TransitionKey.REWARD.value] == 0.0
    assert out[TransitionKey.DONE.value] is False


def test_sub_threshold_does_not_update():
    step = _FakeRewardStep(fixed_reward=0.4)
    t = _trans()
    out = step(t)
    assert out[TransitionKey.REWARD.value] == 0.0
    assert out[TransitionKey.DONE.value] is False
    assert "reward_classifier_frequency" in out[TransitionKey.INFO.value]


def test_success_reward_and_termination():
    step = _FakeRewardStep(fixed_reward=1.0, terminate_on_success=True)
    out = step(_trans())
    assert out[TransitionKey.REWARD.value] == pytest.approx(2.5)
    assert out[TransitionKey.DONE.value] is True


def test_success_without_termination():
    step = _FakeRewardStep(fixed_reward=1.0, terminate_on_success=False)
    out = step(_trans())
    assert out[TransitionKey.REWARD.value] == pytest.approx(2.5)
    assert out[TransitionKey.DONE.value] is False


def test_dispatch_none_or_manual():
    assert build_reward_model_step(None) is None
    assert build_reward_model_step({}) is None
    assert build_reward_model_step({"type": "manual"}) is None
    assert build_reward_model_step({"type": "none"}) is None


def test_dispatch_height_gripper():
    step = build_reward_model_step(
        {"type": "height_gripper", "z_index": 14, "gripper_index": 17, "success_reward": 5.0}
    )
    assert isinstance(step, HeightGripperRewardStep)
    assert step.config.z_index == 14
    assert step.config.gripper_index == 17
    assert step.config.success_reward == 5.0


def test_dispatch_cnn_no_path():
    step = build_reward_model_step({"type": "cnn", "pretrained_path": None})
    assert isinstance(step, CNNRewardProcessorStep)
    assert step._classifier is None


def test_dispatch_sarm_no_path():
    step = build_reward_model_step(
        {"type": "sarm", "pretrained_path": None, "task": "lift", "reward_mode": "delta"}
    )
    assert isinstance(step, SARMRewardProcessorStep)
    assert step._model is None
    assert step.config.reward_mode == "delta"


def test_dispatch_bad_type():
    with pytest.raises(ValueError):
        build_reward_model_step({"type": "bogus"})


def test_dispatch_bad_reward_mode():
    with pytest.raises(ValueError):
        build_reward_model_step({"type": "sarm", "reward_mode": "bogus"})


def test_dispatch_extra_keys_are_filtered():
    step = build_reward_model_step(
        {
            "type": "height_gripper",
            "task": "junk",
            "stats_dataset_repo_id": "junk",
            "z_index": 5,
        }
    )
    assert isinstance(step, HeightGripperRewardStep)
    assert step.config.z_index == 5


def test_dispatch_respects_terminate_on_success():
    step_term = build_reward_model_step({"type": "cnn", "terminate_on_success": True})
    step_noterm = build_reward_model_step({"type": "cnn", "terminate_on_success": False})
    assert step_term.terminate_on_success is True
    assert step_noterm.terminate_on_success is False
