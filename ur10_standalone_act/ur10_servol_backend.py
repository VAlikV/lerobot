"""
Thread-based servoL position-control backend for the standalone UR10e follower.

This is the NON-compliant counterpart to `ur10_osc_controller.UR10OSCController`.
For free-space motion (pick / place / transport / jig handling) you do NOT want the
torque-mode compliance of OSC — you want UR's own stiff, well-tuned, full-dynamics
position controller. servoL gives exactly that, with none of the Jacobian-transpose
OSC orientation tilt/vibration.

It exposes the SAME interface as `UR10OSCController` so `UR10Follower` can swap
between them with a single `control_backend` flag, and the recorded dataset / ACT
training are byte-identical across backends (same `{x,y,z,yaw,gripper}.pos` action,
same 11-D observation). Only the press stage needs OSC; everything else uses this.

Architecture (lifted from the battle-tested `src/lerobot/robots/ur10/ur10_robot.py`,
trimmed of gym / cameras / HIL baselines):
  - A background thread streams `servoL(target, ...)` at ~200 Hz. servoL self-paces
    in ur_rtde's C++ layer (blocks for `dt`), so the loop needs no extra sleep and the
    URScript watchdog never fires across idle gaps.
  - `set_target()` is a sub-microsecond locked write of the shared target; the stream
    thread picks it up on its next tick. NO per-call servo entry point.
  - Reset uses a blocking `moveL` (speed/accel limited) — pauses the stream, moves,
    updates the target, resumes. A far servoL jump would move at servo speed (unsafe
    for a reset); moveL is bounded.
  - servoStop is known to WEDGE against this controller after a few cycles, so every
    blocking ctrl call that can wedge runs under a hard deadline that force-resets the
    RTDE handle on hang (see `_ctrl_call_with_deadline`).

NOTE: servoL and OSC's directTorque are mutually exclusive control modes — only one
backend may be connected to the robot at a time. Switching happens BETWEEN behaviour-
tree stages, never mid-episode.

Requirements: UR10e e-Series, ur_rtde, numpy, scipy. Run with the `lerobot` conda env.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class UR10ServoLBackend:
    """servoL position-control backend with the `UR10OSCController` method surface.

    Implements: start/stop, is_alive, get_current_tcp, get_state, set_target,
    move_to_pose. The follower treats this and the OSC controller interchangeably.
    """

    def __init__(self, robot_ip,
                 frequency=500,
                 tcp_offset=None,
                 set_payload=False, payload_mass=1.3, payload_cog=(0.0, 0.0, 0.0),
                 gripper_port="/dev/ttyACM0", gripper_baudrate=115200, use_gripper=False,
                 joints_init=None, joints_init_speed=1.05,
                 stream_frequency_hz=200,
                 servo_lookahead_time=0.15, servo_gain=100.0,
                 reset_speed=0.1, reset_acceleration=0.1,
                 launch_timeout=5.0, verbose=False):
        self.robot_ip = robot_ip
        self.frequency = int(frequency)

        # TCP / payload. Default: trust the pendant (set_payload=False -> no RTDE override),
        # matching the OSC backend so gravity comp stays consistent across a mode switch.
        self.tcp_offset = list(tcp_offset) if tcp_offset is not None else [0.0] * 6
        self.set_payload = bool(set_payload)
        self.payload_mass = float(payload_mass)
        self.payload_cog = list(payload_cog)

        self.gripper_port = gripper_port
        self.gripper_baudrate = gripper_baudrate
        self.use_gripper = bool(use_gripper)

        self.joints_init = None if joints_init is None else list(joints_init)
        self.joints_init_speed = float(joints_init_speed)

        self.stream_dt = 1.0 / max(1, int(stream_frequency_hz))
        self.servo_lookahead_time = float(servo_lookahead_time)
        self.servo_gain = float(servo_gain)
        self.reset_speed = float(reset_speed)
        self.reset_acceleration = float(reset_acceleration)
        self.launch_timeout = float(launch_timeout)
        self.verbose = bool(verbose)

        self.rtde_c = None
        self.rtde_r = None
        self.gripper = None
        self._connected = False

        # Streaming-thread state.
        self._target = None                  # shared list[6] absolute TCP target
        self._target_lock = threading.Lock()
        self._ctrl_lock = threading.Lock()   # serialize all rtde_c calls (NOT thread-safe)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._stream_failed = threading.Event()
        self._stream_thread = None

        # Gripper command tracking (send only on transition; keep serial off the hot path).
        self._gripper_closed = False

        # Latest telemetry snapshot (published by the stream thread; read lock-free here is
        # fine because get_state copies under no contention with servoL's own socket).
        self._safety = 0                     # 0=ok 1=protective 2=estop

    # ---- lifecycle ---------------------------------------------------------
    def start(self, wait=True):
        import rtde_control
        import rtde_receive

        logger.info("[servol] connecting UR10e @ %s (%d Hz) ...", self.robot_ip, self.frequency)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip, float(self.frequency))
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip, float(self.frequency))

        with self._ctrl_lock:
            if self.set_payload:
                self.rtde_c.setTcp(list(self.tcp_offset))
                self.rtde_c.setPayload(self.payload_mass, self.payload_cog)
            else:
                logger.info("[servol] using pendant payload/TCP (set_payload=False).")

        if self.use_gripper:
            from rc10_api.gripper import Gripper
            self.gripper = Gripper(device=self.gripper_port, baudrate=self.gripper_baudrate)

        if self.joints_init is not None:
            logger.info("[servol] moveJ to init joints %s ...", self.joints_init)
            with self._ctrl_lock:
                self.rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)

        # Initialise the streaming target to the live TCP so the first servoL is a no-op.
        with self._target_lock:
            self._target = list(self.rtde_r.getActualTCPPose())

        self._connected = True
        self._stop_event.clear()
        self._pause_event.clear()
        self._stream_failed.clear()
        self._stream_thread = threading.Thread(target=self._stream_loop, name="UR10ServoLStream",
                                               daemon=True)
        self._stream_thread.start()
        logger.info("[servol] streaming thread started (dt=%.4fs, lookahead=%.3fs, gain=%.0f).",
                    self.stream_dt, self.servo_lookahead_time, self.servo_gain)

    def stop(self, wait=True):
        # Stop the stream thread first so it can't race servoStop / stopScript.
        self._stop_event.set()
        self._pause_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        if self.rtde_c is not None:
            self._ctrl_call_with_deadline("servoStop", lambda: self.rtde_c.servoStop(10.0), 12.0)
            self._ctrl_call_with_deadline("stopScript", lambda: self.rtde_c.stopScript(), 5.0)
            try:
                self.rtde_c.disconnect()
            except Exception:
                logger.exception("[servol] rtde_c.disconnect failed")
        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
            except Exception:
                logger.exception("[servol] rtde_r.disconnect failed")
        if self.gripper is not None:
            try:
                self.gripper.close()
            except Exception:
                pass
        self._connected = False
        logger.info("[servol] backend stopped.")

    def is_alive(self) -> bool:
        return (self._connected
                and self._stream_thread is not None
                and self._stream_thread.is_alive()
                and not self._stream_failed.is_set())

    @property
    def is_ready(self) -> bool:
        return self.is_alive()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ---- telemetry / API (mirrors UR10OSCController) -----------------------
    def get_current_tcp(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualTCPPose(), dtype=float)

    def get_state(self) -> dict:
        """Latest robot state as a dict — same keys the OSC controller publishes.

        `rtde_r` is a separate socket from the stream thread's `rtde_c`, so these reads
        don't contend with servoL.
        """
        q = np.array(self.rtde_r.getActualQ(), dtype=float)
        qd = np.array(self.rtde_r.getActualQd(), dtype=float)
        tcp = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
        # Poll safety here (main thread) rather than in the stream thread: RTDEReceive is
        # read from one thread only, matching the proven UR10Robot model. The stream
        # thread touches rtde_c exclusively.
        try:
            if self.rtde_r.isEmergencyStopped():
                self._safety = 2
            elif self.rtde_r.isProtectiveStopped():
                self._safety = 1
            else:
                self._safety = 0
        except Exception:
            pass
        return {"q": q, "qd": qd, "tcp": tcp, "stamp": time.time(),
                "dhat": 0.0, "safety": int(self._safety)}

    def set_target(self, pose, close_gripper=False):
        """Update the streamed absolute TCP target [x,y,z,rx,ry,rz]. Non-blocking."""
        pose = np.asarray(pose, dtype=float).reshape(6)
        with self._target_lock:
            self._target = list(pose)
        self._apply_gripper(close_gripper)
        return pose

    def move_to_pose(self, pose, close_gripper=False):
        """Blocking, speed/accel-limited reset move via moveL.

        Pauses the stream, exits servoL mode, moveL to the pose, updates the shared
        target so the resumed thread holds there (no snap back to the old target), then
        resumes. Mirrors `UR10Robot.move_to_pose`.
        """
        pose = list(np.asarray(pose, dtype=float).reshape(6))
        streaming = self._stream_thread is not None and self._stream_thread.is_alive()
        if streaming:
            self._pause_event.set()
            # servoStop is required: moveL does not implicitly leave servoL mode on this
            # firmware, so without it the moveL is silently dropped. Deadline-guarded
            # because servoStop wedges after a few cycles.
            self._ctrl_call_with_deadline("servoStop", lambda: self.rtde_c.servoStop(10.0), 2.0)
        try:
            with self._ctrl_lock:
                self.rtde_c.moveL(pose, self.reset_speed, self.reset_acceleration, False)
            with self._target_lock:
                self._target = list(pose)
        finally:
            if streaming:
                self._pause_event.clear()
        self._apply_gripper(close_gripper)
        return np.asarray(pose, dtype=float)

    # ---- internals ---------------------------------------------------------
    def _apply_gripper(self, close_gripper: bool) -> None:
        """Send the gripper command only on a state transition (serial off the hot path)."""
        if self.gripper is None:
            return
        close = bool(close_gripper)
        if close and not self._gripper_closed:
            self.gripper.send(-1); self._gripper_closed = True
        elif not close and self._gripper_closed:
            self.gripper.send(1); self._gripper_closed = False

    def _stream_loop(self) -> None:
        """Background loop: stream servoL to the latest target until stopped.

        servoL blocks for `stream_dt` in C++, self-pacing the loop. This thread touches
        rtde_c ONLY (safety polling lives in get_state on the main thread) so there is no
        concurrent access to the RTDEReceive socket.
        """
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._stop_event.wait(timeout=0.01)
                continue

            with self._target_lock:
                pose = None if self._target is None else list(self._target)
            if pose is None:
                self._stop_event.wait(timeout=self.stream_dt)
                continue

            try:
                with self._ctrl_lock:
                    self.rtde_c.servoL(pose, 0.0, 0.0, self.stream_dt,
                                       self.servo_lookahead_time, self.servo_gain)
            except Exception:
                logger.exception("[servol] streaming servoL failed; pausing thread")
                self._stream_failed.set()
                self._pause_event.set()
                continue

    def _ctrl_call_with_deadline(self, name: str, fn, deadline_s: float) -> None:
        """Run a blocking rtde_c call on a side thread with a hard deadline.

        On hang: force-disconnect rtde_c (unwedges the C++ call), reconnect a fresh
        handle + a fresh ctrl lock, reapply TCP/payload. If recovery fails, flag the
        stream unhealthy so `is_alive()` flips. Trimmed from `UR10Robot`.
        """
        done = threading.Event()
        err: list[BaseException] = []

        def target() -> None:
            try:
                with self._ctrl_lock:
                    fn()
            except BaseException as e:  # noqa: BLE001
                err.append(e)
            finally:
                done.set()

        t = threading.Thread(target=target, name=f"servol-{name}", daemon=True)
        t.start()
        if done.wait(deadline_s):
            if err:
                logger.exception("[servol] %s failed", name, exc_info=err[0])
            return

        logger.warning("[servol] %s wedged for %.1fs; replacing rtde_c + ctrl_lock", name, deadline_s)
        try:
            self.rtde_c.disconnect()
        except Exception:
            logger.exception("[servol] old rtde_c.disconnect failed (continuing)")
        try:
            import rtde_control
            new_ctrl = rtde_control.RTDEControlInterface(self.robot_ip, float(self.frequency))
            new_lock = threading.Lock()
            with new_lock:
                if self.set_payload:
                    new_ctrl.setTcp(list(self.tcp_offset))
                    new_ctrl.setPayload(self.payload_mass, self.payload_cog)
            self.rtde_c = new_ctrl
            self._ctrl_lock = new_lock
            logger.info("[servol] rtde_c + ctrl_lock replaced after wedge")
        except Exception:
            logger.exception("[servol] rtde_c reconnect failed; flagging stream unhealthy")
            self._stream_failed.set()
