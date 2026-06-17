"""Relative-target gamepad teleoperator for the standard lerobot `record_loop`.

PURE MOTION. `record_loop` records whatever `teleop.get_action()` returns and
forwards it to `robot.send_action()`. This teleop turns gamepad sticks into the
RELATIVE-to-home target our UR10Follower expects —
`{x.pos, y.pos, z.pos, yaw.pos, gripper.pos}` — by accumulating deltas (R1 deadman)
into a latched relative target. So the recorded action is the relative target
(not raw deltas), exactly as we want.

Episode control (Triangle/Square/Cross) is NOT handled here. It lives in the
background `GamepadListener` thread, which sets `record_loop`'s `events` dict
asynchronously — the same pattern as `init_keyboard_listener()`. That is what
fixes the old in-loop "press success twice" / "1-frame episode" bugs: button
reading no longer happens at the (possibly slow) record-loop rate.

This teleop reads LATCHED gamepad state from the listener; it never touches pygame
itself, so there is exactly one pygame consumer in the process.

Clipping: if `teleop.robot` is set, the relative target is clipped so home+rel
stays inside the robot's ee_bounds/yaw bounds — keeping the RECORDED action equal
to what robot.send_action() actually applies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

from gamepad_listener import GamepadListener

logger = logging.getLogger(__name__)


@dataclass
class RelativeGamepadTeleopConfig(TeleoperatorConfig):
    ee_step: tuple[float, float, float] = (0.001, 0.001, 0.001)  # m per unit-action
    yaw_step: float = 0.006                                       # rad per unit-action
    deadzone: float = 0.05                                        # stick deadzone
    use_yaw: bool = True
    use_gripper: bool = True
    invert_delta_x: bool = True
    invert_delta_y: bool = True
    invert_delta_z: bool = False
    invert_delta_yaw: bool = False
    stick_cal_s: float = 1.5
    fps: int = 30
    poll_hz: int = 100


class RelativeGamepadTeleop(Teleoperator):
    config_class = RelativeGamepadTeleopConfig
    name = "relative_gamepad"

    def __init__(self, config: RelativeGamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self._listener: GamepadListener | None = None
        self.robot = None         # set externally: used to clip rel to ee_bounds/home
        self.events: dict | None = None  # the listener's events dict (set on connect)
        self._ee_step = np.array(config.ee_step, dtype=np.float32)
        self._rel = np.zeros(3, dtype=np.float32)
        self._rel_yaw = 0.0
        self._bias = np.zeros(4, dtype=np.float32)
        self.n_calls = 0   # get_action calls since last reset() == loop iterations

    # -- Teleoperator interface --------------------------------------------
    @property
    def action_features(self) -> dict:
        return {"x.pos": float, "y.pos": float, "z.pos": float,
                "yaw.pos": float, "gripper.pos": float}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._listener is not None and self._listener.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        self._listener = GamepadListener(
            deadzone=self.config.deadzone, use_yaw=self.config.use_yaw,
            poll_hz=self.config.poll_hz,
            invert_x=self.config.invert_delta_x, invert_y=self.config.invert_delta_y,
            invert_z=self.config.invert_delta_z, invert_yaw=self.config.invert_delta_yaw,
        )
        self._listener.start_and_wait()
        self.events = self._listener.events
        self._calibrate_bias()

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    # -- relative-target driving -------------------------------------------
    def _calibrate_bias(self) -> None:
        """Sample resting-stick bias (hands off) so a biased stick doesn't creep."""
        n = max(1, int(self.config.stick_cal_s * self.config.fps))
        dt = 1.0 / self.config.fps
        logger.info("[relative-gamepad] release sticks; calibrating bias %.1fs ...",
                    self.config.stick_cal_s)
        acc = np.zeros(4, dtype=np.float32)
        for _ in range(n):
            acc += self._listener.get_deltas()
            precise_sleep(dt)
        self._bias = acc / n
        logger.info("[relative-gamepad] bias = %s", self._bias.round(3).tolist())

    def _dz(self, v: float) -> float:
        return 0.0 if abs(v) < self.config.deadzone else float(v)

    def reset(self) -> None:
        """Zero the relative accumulator (call at episode start, after capture_home /
        go_to_home). Sync the latched gripper to the robot's actual state so we neither
        drop a gripped object nor immediately re-close one that go_to_home just opened."""
        self._rel = np.zeros(3, dtype=np.float32)
        self._rel_yaw = 0.0
        self.n_calls = 0
        if self.robot is not None and self._listener is not None:
            self._listener.gripper_open = bool(self.robot.gripper_is_open)

    def get_action(self) -> dict:
        self.n_calls += 1
        d = self._listener.get_deltas()        # latched [dx, dy, dz, dyaw]
        if self._listener.should_intervene():  # R1 deadman held
            self._rel = self._rel + np.array(
                [self._dz(d[0] - self._bias[0]), self._dz(d[1] - self._bias[1]),
                 self._dz(d[2] - self._bias[2])], dtype=np.float32) * self._ee_step
            self._rel_yaw += self._dz(d[3] - self._bias[3]) * self.config.yaw_step

        # Clip rel so home+rel stays in bounds (recorded == what send_action applies).
        if self.robot is not None:
            self._rel = np.clip(self._rel, self.robot.ee_min - self.robot.home_xyz,
                                self.robot.ee_max - self.robot.home_xyz).astype(np.float32)
            self._rel_yaw = float(np.clip(self._rel_yaw, self.robot.config.yaw_min,
                                          self.robot.config.yaw_max))

        gripper_open = self._listener.gripper_open if self.config.use_gripper else True
        return {
            "x.pos": float(self._rel[0]), "y.pos": float(self._rel[1]), "z.pos": float(self._rel[2]),
            "yaw.pos": float(self._rel_yaw), "gripper.pos": 1.0 if gripper_open else 0.0,
        }

    def get_teleop_events(self) -> dict[str, Any]:
        # Not used by record_loop, but provided for completeness.
        if self._listener is None:
            return {}
        return {TeleopEvents.IS_INTERVENTION: bool(self._listener.should_intervene())}
