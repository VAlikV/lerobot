"""Unit tests for SARMRewardProcessorStep reward modes + ring buffer.

Uses a stub SARMRewardModel so tests don't require a trained checkpoint or CLIP.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest
import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.reward_model.sarm import (
    SARMRewardConfig,
    SARMRewardProcessorStep,
)


def _make_step(reward_mode: str = "binary", terminate_on_success: bool = True) -> SARMRewardProcessorStep:
    cfg = SARMRewardConfig(
        pretrained_path=None,
        reward_mode=reward_mode,
        success_threshold=0.8,
        success_reward=1.0,
    )
    step = SARMRewardProcessorStep(config=cfg, terminate_on_success=terminate_on_success)

    # Manually plant a stub so __call__ uses the async-submit path.
    step._image_key = "observation.images.front"
    step._state_key = "observation.state"
    step._delta_indices = [-2, -1, 0, 1, 2]
    step._center_idx = 2
    step._image_buf = deque(maxlen=3)
    step._state_buf = deque(maxlen=3)

    class _StubModel:
        pass

    step._model = _StubModel()
    step._preprocess = None
    return step


def _trans(image: torch.Tensor | None, state: torch.Tensor | None, done: bool = False):
    obs = {}
    if image is not None:
        obs["observation.images.front"] = image
    if state is not None:
        obs["observation.state"] = state
    return {
        TransitionKey.OBSERVATION.value: obs,
        TransitionKey.REWARD.value: 0.0,
        TransitionKey.DONE.value: done,
        TransitionKey.TRUNCATED.value: False,
        TransitionKey.INFO.value: {},
    }


def test_ring_buffer_push_grows_then_slides():
    step = _make_step()
    img = torch.zeros(3, 8, 8)
    state = torch.zeros(10)
    for i in range(5):
        step._push_obs_to_buffer({"observation.images.front": img + i, "observation.state": state + i})
    assert len(step._image_buf) == 3  # maxlen
    assert step._image_buf[-1][0, 0, 0] == 4  # most recent


def test_push_missing_image_returns_false():
    step = _make_step()
    ok = step._push_obs_to_buffer({"observation.state": torch.zeros(10)})
    assert ok is False


def test_window_builds_with_replicated_current_for_future_slots(monkeypatch):
    step = _make_step()
    img = torch.arange(3 * 2 * 2).reshape(3, 2, 2).float()
    state = torch.arange(5).float()
    step._push_obs_to_buffer({"observation.images.front": img, "observation.state": state})
    snap_imgs, snap_states = step._snapshot_buffers()
    stacked_img, stacked_state = step._build_window_from_snapshot(snap_imgs, snap_states)
    # 5 delta indices, current replicated everywhere since buf has only 1 frame.
    assert stacked_img.shape[0] == 5
    assert stacked_state.shape[0] == 5
    # Every frame must equal the current (since no past).
    assert torch.allclose(stacked_img[0], img)
    assert torch.allclose(stacked_img[-1], img)


def test_reset_clears_buffers_and_counters():
    step = _make_step()
    step._push_obs_to_buffer(
        {"observation.images.front": torch.zeros(3, 4, 4), "observation.state": torch.zeros(10)}
    )
    step._step_counter = 42
    step._prev_progress = 0.3
    step._last_progress = 0.5
    step.reset()
    assert len(step._image_buf) == 0
    assert step._step_counter == 0
    assert step._prev_progress == 0.0
    assert step._last_progress == 0.0


def _patch_progress_series(step, series: list[float]) -> None:
    """Make __call__ observe progress=series[i] at iteration i (no async lag).

    The real step submits compute jobs to a background thread. For tests we want
    deterministic per-step values. We run _compute_progress_from_buffer
    synchronously inside the fake submit, write the result directly to
    _last_progress, and return None so ``_pending_future`` stays None — which
    lets the next step's ``submit`` path fire again.
    """
    step._pending_future = None
    it = iter(series)

    def _fake(*a: Any, **kw: Any) -> float:
        return next(it)

    step._compute_progress_from_buffer = _fake  # type: ignore[assignment]

    def _submit_sync(fn, *args, **kwargs):
        step._last_progress = fn(*args, **kwargs)
        return None

    step._executor.submit = _submit_sync  # type: ignore[method-assign]


def test_delta_mode_telescopes_to_1(monkeypatch):
    step = _make_step(reward_mode="delta")
    n = 10
    series = [i / (n - 1) for i in range(n)]  # 0.0 → 1.0 linearly (hits threshold at 0.888…)
    _patch_progress_series(step, series + [1.0])  # extra terminal call

    img = torch.zeros(3, 4, 4)
    state = torch.zeros(10)
    total_reward = 0.0
    terminated_at = None
    for t in range(n):
        out = step(_trans(img, state))
        total_reward += float(out[TransitionKey.REWARD.value])
        if out[TransitionKey.DONE.value]:
            terminated_at = t
            break
    # delta returns should ≈ final progress at termination.
    assert terminated_at is not None
    assert total_reward == pytest.approx(series[terminated_at], abs=1e-3)


def test_binary_mode_only_fires_on_threshold():
    step = _make_step(reward_mode="binary")
    series = [0.1, 0.2, 0.5, 0.7, 0.95]
    _patch_progress_series(step, series)
    img = torch.zeros(3, 4, 4)
    state = torch.zeros(10)
    rewards = []
    dones = []
    for _ in range(5):
        out = step(_trans(img, state))
        rewards.append(float(out[TransitionKey.REWARD.value]))
        dones.append(bool(out[TransitionKey.DONE.value]))
        if out[TransitionKey.DONE.value]:
            break
    # last reward should be success_reward=1.0 after crossing 0.8
    assert rewards[-1] == pytest.approx(1.0)
    assert dones[-1] is True
    # earlier rewards should be 0 (default carried through)
    assert all(r == 0.0 for r in rewards[:-1])


def test_dense_mode_writes_raw_progress_every_step():
    step = _make_step(reward_mode="dense", terminate_on_success=False)
    series = [0.1, 0.3, 0.55]
    _patch_progress_series(step, series)
    img = torch.zeros(3, 4, 4)
    state = torch.zeros(10)
    actual = []
    for _ in range(3):
        out = step(_trans(img, state))
        actual.append(float(out[TransitionKey.REWARD.value]))
    assert actual == pytest.approx(series, abs=1e-3)


def test_progress_written_to_info():
    step = _make_step(reward_mode="dense", terminate_on_success=False)
    _patch_progress_series(step, [0.42])
    out = step(_trans(torch.zeros(3, 4, 4), torch.zeros(10)))
    assert out[TransitionKey.INFO.value]["sarm_progress"] == pytest.approx(0.42)
