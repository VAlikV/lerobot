"""Tests for the gamepad stage-advance button plumbing.

We do NOT require a real gamepad. We exercise `GamepadController.update()`
with a mocked pygame event queue (sequence of JOYBUTTONDOWN events) and
verify the consume_stage_advance() latch behaviour.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeJoy:
    def get_button(self, i):  # always not-pressed
        return False

    def get_axis(self, i):
        return 0.0


class _FakeEvent:
    def __init__(self, etype, button=None):
        self.type = etype
        if button is not None:
            self.button = button


def _install_fake_pygame(events):
    """Install a minimal pygame module into sys.modules.

    `events` is a list of _FakeEvent. Subsequent calls to pygame.event.get()
    drain the list (first call returns all events, second returns []).
    """
    fake = types.ModuleType("pygame")
    fake.JOYBUTTONDOWN = 10
    fake.JOYBUTTONUP = 11

    class _ErrType(Exception):
        pass

    fake.error = _ErrType

    state = {"drained": False}

    def _get_events():
        if state["drained"]:
            return []
        state["drained"] = True
        return events

    fake.event = types.SimpleNamespace(get=_get_events)
    sys.modules["pygame"] = fake
    return fake


@pytest.fixture(autouse=True)
def _restore_pygame():
    orig = sys.modules.get("pygame")
    yield
    if orig is None:
        sys.modules.pop("pygame", None)
    else:
        sys.modules["pygame"] = orig


def _make_controller(button_idx):
    from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController

    ctrl = GamepadController(stage_advance_button=button_idx)
    ctrl.joystick = _FakeJoy()
    return ctrl


def test_consume_stage_advance_latches_once_per_press():
    _install_fake_pygame([_FakeEvent(10, button=0)])  # JOYBUTTONDOWN, button 0
    ctrl = _make_controller(button_idx=0)
    ctrl.update()
    assert ctrl.consume_stage_advance() is True
    # Second consume with no new events → False (read-and-clear).
    assert ctrl.consume_stage_advance() is False


def test_non_matching_button_does_not_trigger():
    _install_fake_pygame([_FakeEvent(10, button=2)])  # Triangle-ish, not ours
    ctrl = _make_controller(button_idx=0)
    ctrl.update()
    assert ctrl.consume_stage_advance() is False


def test_multiple_presses_collapse_to_single_pending():
    # JOYBUTTONDOWN fires once per physical press; pygame does NOT auto-repeat
    # while held. But if user mashes it twice within one update cycle, both
    # events arrive in the queue. _stage_advance_pending is a bool, so they
    # collapse — this matches the user's intent (one advance per update tick,
    # matching the consumer's single-shot consume).
    _install_fake_pygame([_FakeEvent(10, button=0), _FakeEvent(10, button=0)])
    ctrl = _make_controller(button_idx=0)
    ctrl.update()
    assert ctrl.consume_stage_advance() is True
    assert ctrl.consume_stage_advance() is False


def test_teleop_get_events_emits_stage_advance():
    _install_fake_pygame([_FakeEvent(10, button=0)])

    from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
    from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
    from lerobot.teleoperators.utils import TeleopEvents

    teleop = GamepadTeleop(GamepadTeleopConfig(stage_advance_button=0))
    # Bypass `connect()` (which would init real pygame) — inject the fake.
    teleop.gamepad = _make_controller(button_idx=0)

    events = teleop.get_teleop_events()
    assert events[TeleopEvents.STAGE_ADVANCE] is True
    # Second call: latch cleared.
    events2 = teleop.get_teleop_events()
    assert events2[TeleopEvents.STAGE_ADVANCE] is False
