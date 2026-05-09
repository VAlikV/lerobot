"""Unit tests for StageAnnotatorProcessorStep."""

from __future__ import annotations

from lerobot.processor.core import TransitionKey
from lerobot.processor.stage_annotator import StageAnnotatorProcessorStep
from lerobot.teleoperators.utils import TeleopEvents


def _tick(step, advance: bool = False):
    t = {
        TransitionKey.OBSERVATION.value: {},
        TransitionKey.ACTION.value: None,
        TransitionKey.REWARD.value: 0.0,
        TransitionKey.DONE.value: False,
        TransitionKey.TRUNCATED.value: False,
        TransitionKey.INFO.value: {TeleopEvents.STAGE_ADVANCE: advance},
    }
    return step(t)


def test_empty_stage_names_is_noop():
    step = StageAnnotatorProcessorStep(stage_names=[])
    out = _tick(step, advance=True)
    # No stage_index / stage_name keys added.
    info = out[TransitionKey.INFO.value]
    assert "stage_index" not in info
    assert "stage_name" not in info
    # And flush returns all None.
    assert step.flush_episode_annotation() == (None, None, None)


def test_reset_seeds_stage_zero():
    step = StageAnnotatorProcessorStep(stage_names=["a", "b", "c"])
    step.reset()
    out = _tick(step, advance=False)
    info = out[TransitionKey.INFO.value]
    assert info["stage_index"] == 0
    assert info["stage_name"] == "a"


def test_single_advance_records_next_stage_start():
    step = StageAnnotatorProcessorStep(stage_names=["a", "b", "c"])
    step.reset()
    # frame 0: no advance. frame_counter becomes 1.
    _tick(step, advance=False)
    # frame 1: advance. stage b starts at frame 1.
    out = _tick(step, advance=True)
    info = out[TransitionKey.INFO.value]
    assert info["stage_index"] == 1
    assert info["stage_name"] == "b"
    assert info["stage_started_this_frame"] == 1


def test_multi_advance_in_one_tick_advances_all():
    step = StageAnnotatorProcessorStep(stage_names=["a", "b", "c"])
    step.reset()
    # Simulate two advances landing on the same tick by hand: the flag is
    # a bool, so in this test we pump two ticks each with advance=True.
    _tick(step, advance=True)
    _tick(step, advance=True)
    # After 2 advances + 1 seeded: current stage should be index 2.
    assert step.current_stage == 2


def test_advance_past_last_is_ignored():
    step = StageAnnotatorProcessorStep(stage_names=["a", "b"])
    step.reset()
    _tick(step, advance=True)       # → stage 1 (b)
    _tick(step, advance=True)       # ignored (already at last)
    assert step.current_stage == 1


def test_flush_monotonic_and_ends_correct():
    step = StageAnnotatorProcessorStep(stage_names=["a", "b", "c"])
    step.reset()
    _tick(step, advance=False)      # frame 0 in stage a
    _tick(step, advance=False)      # frame 1 in stage a
    _tick(step, advance=True)       # frame 2: enters b
    _tick(step, advance=False)      # frame 3 in stage b
    _tick(step, advance=True)       # frame 4: enters c
    _tick(step, advance=False)      # frame 5 in stage c

    # Reached final stage + episode succeeded → all stages kept.
    names, starts, ends = step.flush_episode_annotation(episode_succeeded=True)
    assert names == ["a", "b", "c"]
    assert starts == [0, 2, 4]
    # ends[k] = starts[k+1] - 1; last ends at frame_counter - 1 (= 5).
    assert ends == [1, 3, 5]
    assert step.extension_frame_count() == 0


def test_flush_empty_when_no_ticks():
    step = StageAnnotatorProcessorStep(stage_names=["a", "b"])
    # NOT reset (so _stage_starts is empty), no ticks → None triple.
    names, starts, ends = step.flush_episode_annotation()
    assert names is None and starts is None and ends is None


def test_flush_partial_drops_last_stage_and_extends_prev():
    """Operator advanced K<N-1 times → last-entered stage is partial. It is
    dropped from the annotation and the previous stage's end is extended
    to cover partial-stage frames. Works regardless of episode_succeeded,
    because final stage was never reached.
    """
    step = StageAnnotatorProcessorStep(stage_names=["a", "b", "c"])
    step.reset()
    _tick(step, advance=False)      # frame 0, stage a
    _tick(step, advance=False)      # frame 1, stage a
    _tick(step, advance=True)       # frame 2, enters b (partial — no next press)
    _tick(step, advance=False)      # frame 3, stage b
    _tick(step, advance=False)      # frame 4, stage b

    # Even with episode_succeeded=True, partial because never reached "c".
    names, starts, ends = step.flush_episode_annotation(episode_succeeded=True)
    assert names == ["a"]
    assert starts == [0]
    # "a" extended from [0,1] to [0, frame_counter-1=4] to swallow partial b.
    assert ends == [4]
    # 3 frames of stage b (frames 2,3,4) folded into the extension.
    assert step.extension_frame_count() == 3


def test_flush_reached_final_without_success_drops_final():
    """Operator reached the final configured stage but env/operator did NOT
    signal success → final stage treated as partial (dropped + prev
    extended). This is variant B of the partial-stage spec.
    """
    step = StageAnnotatorProcessorStep(stage_names=["a", "b"])
    step.reset()
    _tick(step, advance=False)      # frame 0, stage a
    _tick(step, advance=True)       # frame 1, enters b (the final stage)
    _tick(step, advance=False)      # frame 2, stage b
    _tick(step, advance=False)      # frame 3, stage b

    names, starts, ends = step.flush_episode_annotation(episode_succeeded=False)
    assert names == ["a"]
    assert starts == [0]
    # "a" extended to frame_counter-1 = 3 to absorb partial b.
    assert ends == [3]
    # 3 frames of b (1,2,3) absorbed.
    assert step.extension_frame_count() == 3

    # Sanity: same trace WITH success → full annotation kept.
    step.reset()
    _tick(step, advance=False)
    _tick(step, advance=True)
    _tick(step, advance=False)
    _tick(step, advance=False)
    names, starts, ends = step.flush_episode_annotation(episode_succeeded=True)
    assert names == ["a", "b"]
    assert starts == [0, 1]
    assert ends == [0, 3]
    assert step.extension_frame_count() == 0


def test_flush_stuck_in_stage_zero_returns_none():
    """No advances at all: entered = [0]. completed = []. Whole episode is
    unclassified (subtask_names=None) → SARM treats as τ=0 everywhere
    (progress=0).
    """
    step = StageAnnotatorProcessorStep(stage_names=["a", "b", "c"])
    step.reset()
    _tick(step, advance=False)
    _tick(step, advance=False)
    _tick(step, advance=False)

    # Regardless of success flag: no completed stages → None triple.
    assert step.flush_episode_annotation(episode_succeeded=False) == (None, None, None)
    assert step.extension_frame_count() == 0

    # Re-run to confirm same result with success=True (stage 0 never
    # actually completed because operator never advanced past it).
    step.reset()
    _tick(step, advance=False)
    _tick(step, advance=False)
    names, starts, ends = step.flush_episode_annotation(episode_succeeded=True)
    # stage 0 is both entered AND the last one → partial under variant B.
    # completed prefix is empty → None.
    assert names is None and starts is None and ends is None


def test_flush_default_success_is_false():
    """Default call (no kwarg) treats episode as NOT-success → applies the
    partial-drop logic. This matters because older callers passing no arg
    now get the stricter behavior.
    """
    step = StageAnnotatorProcessorStep(stage_names=["a", "b"])
    step.reset()
    _tick(step, advance=False)
    _tick(step, advance=True)       # enters b
    _tick(step, advance=False)

    names, starts, ends = step.flush_episode_annotation()
    assert names == ["a"]
    assert ends == [2]
    assert step.extension_frame_count() == 2
