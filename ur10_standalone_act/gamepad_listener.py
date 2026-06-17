"""Background-thread gamepad listener — the SINGLE pygame owner for recording.

This mirrors `init_keyboard_listener()` (lerobot.utils.control_utils): a daemon
thread sets a shared `events` dict ASYNCHRONOUSLY so the standard `record_loop`
just polls `events["exit_early"]` each iteration. Because button reading happens
on its own ~100 Hz thread (NOT inside the record loop), a quick tap is never
missed even when the record loop runs slow — this is what fixes the old
"press success twice" and "1-frame episode" bugs of the in-loop edge detector.

It is also the only place pygame is touched (init + event pump + joystick reads),
so there is no two-thread race on the global pygame event queue. The motion teleop
(`RelativeGamepadTeleop`) reads LATCHED state from here instead of pumping pygame
itself.

PS4 / pygame button map (same as lerobot GamepadController):
    5 = R1 (deadman / intervention, HOLD to drive)
    2 = Triangle (SUCCESS)  -> exit_early           (end episode + save)
    3 = Square   (RERECORD) -> rerecord_episode + exit_early
    1 = Cross    (FAILURE)  -> stop_recording + exit_early
    7 = open gripper (hold)        6 = close gripper (hold)
Axes: 0=left-X, 1=left-Y, 4=right-Y (Z), 3=right-X (yaw).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class GamepadListener(threading.Thread):
    """Daemon thread that owns pygame and publishes gamepad state + episode events."""

    def __init__(
        self,
        deadzone: float = 0.05,
        use_yaw: bool = True,
        poll_hz: int = 100,
        invert_x: bool = True,
        invert_y: bool = True,
        invert_z: bool = False,
        invert_yaw: bool = False,
    ):
        super().__init__(daemon=True)
        self.deadzone = float(deadzone)
        self.use_yaw = bool(use_yaw)
        self.poll_hz = int(poll_hz)
        self._sx = -1.0 if invert_x else 1.0
        self._sy = -1.0 if invert_y else 1.0
        self._sz = -1.0 if invert_z else 1.0
        self._syaw = -1.0 if invert_yaw else 1.0

        # Shared, set by record_loop consumers. Same keys/contract as init_keyboard_listener.
        self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

        # Latched motion state (read by the teleop). Plain attributes — GIL makes
        # single-attribute read/write atomic enough for this use.
        self._deltas = np.zeros(4, dtype=np.float32)   # [dx, dy, dz, dyaw] post-sign/deadzone
        self._intervene = False
        self.gripper_open = True

        self._joy = None
        self._running = False
        self._ready = threading.Event()
        self._err: str | None = None
        # Episode-button edge detection (rising edge = one trigger per physical press).
        # Maintained continuously across episodes so a HELD button never re-fires.
        self._btn_prev = {1: False, 2: False, 3: False}

    # -- lifecycle ----------------------------------------------------------
    def run(self) -> None:
        import os

        # Joystick-only: don't require an X/video display. Without this, pygame.event.pump()
        # raises "video system not initialized" when there's no DISPLAY (e.g. launched over
        # ssh / as a detached process). The dummy driver is harmless when a display exists.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame

        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                self._err = "No gamepad detected."
                self._ready.set()
                return
            self._joy = pygame.joystick.Joystick(0)
            self._joy.init()
            logger.info("[gamepad-listener] %s", self._joy.get_name())
        except Exception as e:  # noqa: BLE001
            self._err = f"pygame init failed: {e}"
            self._ready.set()
            return

        self._running = True
        self._ready.set()
        dt = 1.0 / self.poll_hz
        while self._running:
            t0 = time.perf_counter()
            pygame.event.pump()  # refresh joystick state (sole pump in the process)

            # --- sticks -> signed, deadzoned deltas (same convention as GamepadController +
            #     GamepadTeleop inversions) ---
            ax0 = self._joy.get_axis(0)
            ax1 = self._joy.get_axis(1)
            ax4 = self._joy.get_axis(4)
            ax3 = self._joy.get_axis(3) if self.use_yaw else 0.0
            dz = self.deadzone
            x = 0.0 if abs(ax0) < dz else ax0
            y = 0.0 if abs(ax1) < dz else ax1
            z = 0.0 if abs(ax4) < dz else ax4
            yaw = 0.0 if abs(ax3) < dz else ax3
            self._deltas = np.array(
                [self._sx * x, self._sy * (-y), self._sz * (-z), self._syaw * yaw],
                dtype=np.float32,
            )

            # --- R1 deadman + gripper (level) ---
            self._intervene = bool(self._joy.get_button(5))
            if self._joy.get_button(7):
                self.gripper_open = True
            elif self._joy.get_button(6):
                self.gripper_open = False

            # --- episode buttons (rising-edge -> set events) ---
            cur = {b: bool(self._joy.get_button(b)) for b in (1, 2, 3)}
            if cur[2] and not self._btn_prev[2]:        # Triangle = SUCCESS
                self.events["exit_early"] = True
            elif cur[3] and not self._btn_prev[3]:      # Square = RERECORD
                self.events["rerecord_episode"] = True
                self.events["exit_early"] = True
            elif cur[1] and not self._btn_prev[1]:      # Cross = FAILURE -> stop session
                self.events["stop_recording"] = True
                self.events["exit_early"] = True
            self._btn_prev = cur

            time.sleep(max(dt - (time.perf_counter() - t0), 0.0))

        # teardown (in the same thread that inited pygame)
        try:
            if self._joy is not None:
                self._joy.quit()
            pygame.joystick.quit()
            pygame.quit()
        except Exception:  # noqa: BLE001
            pass

    def start_and_wait(self, timeout: float = 5.0) -> None:
        self.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("gamepad listener did not start in time")
        if self._err is not None:
            raise RuntimeError(self._err)

    def stop(self) -> None:
        self._running = False
        if self.is_alive():
            self.join(timeout=2.0)

    # -- accessors (read by the teleop) ------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._running and self.is_alive()

    def get_deltas(self) -> np.ndarray:
        return self._deltas.copy()

    def should_intervene(self) -> bool:
        return self._intervene
