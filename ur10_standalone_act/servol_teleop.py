"""
Gamepad smoke-test for the servoL position-control backend (`UR10ServoLBackend`).

The servoL counterpart of `ur10_osc_gamepad.py`. It drives the backend DIRECTLY (not
through `UR10Follower`) so you can validate the backend in isolation before recording:

  1. SMOOTHNESS + ORIENTATION HOLD — jog xyz with the gamepad and watch the wrist stay
     pinned during up/down. This is the whole reason for servoL: UR's full-dynamics
     position controller should NOT exhibit the Jacobian-transpose OSC tilt/vibration.
  2. RESET moveL — with `--home_tcp`, the script runs `move_to_pose(home)` at startup
     (speed/accel-limited blocking moveL, then the stream resumes holding home) and
     prints the before/after TCP so you can confirm it lands and doesn't snap back.
  3. GRIPPER — open/close from the gamepad (Cross=close, Triangle/Circle=open per
     GamepadTeleop) goes through `set_target(..., close_gripper)`; the backend owns the
     serial gripper in-process.
  4. SAFETY — the live `safety` field (0=ok, 1=protective, 2=estop) is printed each line,
     so a protective stop is visible the same way record/eval will see it.

Target shaping mirrors `ur10_osc_gamepad.py`: latched `target_xyz` (deltas applied to
the previous command), orientation live during warmup then pinned (optional yaw via
`R_home · R_z(offset)`), R1 (deadman) gates whether deltas accumulate.

Run (lerobot conda env; pendant payload set; Remote Control; e-stop in hand):
    python ur10_standalone_act/servol_teleop.py --ip 192.168.0.100
    python ur10_standalone_act/servol_teleop.py --ip 192.168.0.100 \
        --home_tcp -0.252 -0.563 0.352 3.14159 0 0 --use_yaw --use_gripper
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ur10_servol_backend import UR10ServoLBackend


def main():
    p = argparse.ArgumentParser(description="Gamepad smoke-test for the UR10e servoL backend")
    p.add_argument("--ip", type=str, default="192.168.0.100")
    p.add_argument("--frequency", type=int, default=500)
    # servoL streaming params (conservative defaults — lower gain first when tuning).
    p.add_argument("--stream_frequency_hz", type=int, default=200)
    p.add_argument("--servo_lookahead_time", type=float, default=0.15)
    p.add_argument("--servo_gain", type=float, default=100.0)
    p.add_argument("--reset_speed", type=float, default=0.1)
    p.add_argument("--reset_acceleration", type=float, default=0.1)
    # TCP / payload — trust the pendant by default, matching the follower / OSC backend.
    p.add_argument("--tcp_offset", type=float, nargs=6, default=None)
    p.add_argument("--set_payload", action="store_true", default=False)
    p.add_argument("--payload_mass", type=float, default=1.3)
    p.add_argument("--payload_cog", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--joints_init", type=float, nargs=6, default=None)
    # Reset test: if given, move_to_pose here at startup (blocking moveL), then teleop.
    p.add_argument("--home_tcp", type=float, nargs=6, default=None,
                   help="absolute TCP [x y z rx ry rz]; runs move_to_pose() at startup to test the reset")
    # Teleop shaping (mirrors ur10_osc_gamepad.py).
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--warmup_time_s", type=float, default=5.0)
    p.add_argument("--ee_step_x", type=float, default=0.001)
    p.add_argument("--ee_step_y", type=float, default=0.001)
    p.add_argument("--ee_step_z", type=float, default=0.001)
    p.add_argument("--yaw_step", type=float, default=0.006)
    p.add_argument("--use_yaw", action="store_true", default=False)
    p.add_argument("--invert_delta_x", action="store_true", default=True)
    p.add_argument("--invert_delta_y", action="store_true", default=True)
    p.add_argument("--invert_delta_z", action="store_true", default=False)
    p.add_argument("--invert_delta_yaw", action="store_true", default=False)
    p.add_argument("--use_gripper", action="store_true", default=False)
    p.add_argument("--gripper_port", type=str, default="/dev/ttyACM0")
    p.add_argument("--stick_deadzone", type=float, default=0.05)
    p.add_argument("--stick_cal_s", type=float, default=1.5)
    args = p.parse_args()

    from lerobot.teleoperators.gamepad import GamepadTeleop
    from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
    from lerobot.utils.robot_utils import precise_sleep

    backend = UR10ServoLBackend(
        args.ip, frequency=args.frequency,
        tcp_offset=args.tcp_offset, set_payload=args.set_payload,
        payload_mass=args.payload_mass, payload_cog=args.payload_cog,
        use_gripper=args.use_gripper, gripper_port=args.gripper_port,
        joints_init=args.joints_init,
        stream_frequency_hz=args.stream_frequency_hz,
        servo_lookahead_time=args.servo_lookahead_time, servo_gain=args.servo_gain,
        reset_speed=args.reset_speed, reset_acceleration=args.reset_acceleration,
    )

    print(f"Starting servoL backend (UR10e @ {args.ip}, {args.frequency} Hz) ...")
    backend.start(wait=True)
    print("Backend ready.")

    # --- (2) RESET test: blocking moveL to home, then the stream resumes holding it. ---
    if args.home_tcp is not None:
        before = backend.get_current_tcp()
        print(f"\n[reset-test] move_to_pose -> {np.round(args.home_tcp, 4).tolist()}")
        print(f"[reset-test] TCP before: {np.round(before, 4).tolist()}")
        backend.move_to_pose(args.home_tcp, close_gripper=False)
        time.sleep(0.5)  # let the resumed stream settle on the new target
        after = backend.get_current_tcp()
        err_mm = float(np.linalg.norm(np.array(after[:3]) - np.array(args.home_tcp[:3]))) * 1000.0
        print(f"[reset-test] TCP after : {np.round(after, 4).tolist()}  (landing err {err_mm:.2f} mm)\n")

    teleop = GamepadTeleop(GamepadTeleopConfig(
        use_gripper=args.use_gripper, use_yaw=args.use_yaw,
        invert_delta_x=args.invert_delta_x, invert_delta_y=args.invert_delta_y,
        invert_delta_z=args.invert_delta_z, invert_delta_yaw=args.invert_delta_yaw,
    ))
    teleop.connect()

    ee_step = np.array([args.ee_step_x, args.ee_step_y, args.ee_step_z], dtype=np.float32)
    yaw_step = float(args.yaw_step)
    dt = 1.0 / args.fps
    deadzone = float(args.stick_deadzone)

    # Stick-bias calibration (hands off) — latched accumulation makes any resting bias
    # creep the target.
    def _read_deltas():
        a = teleop.get_action()
        return np.array([
            float(a.get("delta_x", 0.0)), float(a.get("delta_y", 0.0)),
            float(a.get("delta_z", 0.0)),
            float(a.get("delta_yaw", 0.0)) if args.use_yaw else 0.0,
        ], dtype=np.float32)

    n_cal = max(1, int(args.stick_cal_s * args.fps))
    print(f"\n[stick-cal] Release ALL sticks. Measuring resting bias for {args.stick_cal_s:.1f}s ...")
    bias = np.zeros(4, dtype=np.float32)
    for _ in range(n_cal):
        bias += _read_deltas(); precise_sleep(dt)
    bias /= n_cal
    print(f"[stick-cal] resting bias = {bias.round(3).tolist()} (subtracting)")

    def _dz(v):
        return 0.0 if abs(v) < deadzone else float(v)

    target_xyz = np.array(backend.get_current_tcp()[:3], dtype=np.float32)
    fixed_rx = fixed_ry = fixed_rz = None
    R_home = None
    target_yaw_offset = 0.0
    gripper_close = False

    start_t = time.perf_counter()
    warmup_done = False
    max_err_mm = 0.0
    loop_i = 0
    print_every = max(1, args.fps // 10)

    print(f"\n  WARMUP ({args.warmup_time_s}s) — orientation pins to the live wrist at the end.")
    try:
        while True:
            t0 = time.perf_counter()
            if not backend.is_alive():
                print("\n[main] servoL backend not alive — aborting."); break

            a = teleop.get_action()
            enabled = bool(teleop.gamepad.should_intervene())  # R1 deadman
            if enabled:
                dx = _dz(float(a.get("delta_x", 0.0)) - bias[0])
                dy = _dz(float(a.get("delta_y", 0.0)) - bias[1])
                dz = _dz(float(a.get("delta_z", 0.0)) - bias[2])
                dyaw = _dz(float(a.get("delta_yaw", 0.0)) - bias[3]) if args.use_yaw else 0.0
            else:
                dx = dy = dz = dyaw = 0.0
            grip = int(a.get("gripper", 1))
            if grip == 0:
                gripper_close = True
            elif grip == 2:
                gripper_close = False

            tcp = backend.get_current_tcp()
            if enabled:
                target_xyz = target_xyz + np.array([dx, dy, dz], dtype=np.float32) * ee_step

            if fixed_rx is None:
                rx, ry, rz = float(tcp[3]), float(tcp[4]), float(tcp[5])
            elif args.use_yaw and R_home is not None:
                if enabled:
                    target_yaw_offset = float(np.clip(target_yaw_offset + dyaw * yaw_step, -np.pi, np.pi))
                R_t = R_home * Rot.from_euler("z", target_yaw_offset)
                rx, ry, rz = (float(v) for v in R_t.as_rotvec())
            else:
                rx, ry, rz = fixed_rx, fixed_ry, fixed_rz

            backend.set_target(
                [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]), rx, ry, rz],
                close_gripper=gripper_close,
            )

            elapsed = time.perf_counter() - start_t
            if elapsed < args.warmup_time_s:
                print(f"  warmup ... {args.warmup_time_s - elapsed:4.1f}s remaining", end="\r")
            else:
                if not warmup_done:
                    fixed_rx, fixed_ry, fixed_rz = float(tcp[3]), float(tcp[4]), float(tcp[5])
                    R_home = Rot.from_rotvec([fixed_rx, fixed_ry, fixed_rz])
                    target_xyz = np.array(tcp[:3], dtype=np.float32)
                    target_yaw_offset = 0.0
                    max_err_mm = 0.0
                    print("\n  HOLDING TEST — HOLD R1 to drive; release R1 to freeze. Ctrl+C to stop.")
                    warmup_done = True
                err_mm = float(np.linalg.norm(target_xyz - tcp[:3])) * 1000.0
                max_err_mm = max(max_err_mm, err_mm)
                if loop_i % print_every == 0:
                    gate = "R1 DRIVE" if enabled else "hold    "
                    safety = backend.get_state()["safety"]
                    flag = {0: "ok", 1: "PROTECTIVE", 2: "ESTOP"}.get(safety, str(safety))
                    print(f"  [{gate}]  pos_err = {err_mm:6.2f} mm  (max {max_err_mm:6.2f})  "
                          f"grip={'closed' if gripper_close else 'open'}  safety={flag}   ", end="\r")
                    if safety:
                        print("\n[main] SAFETY STOP detected — release on the pendant; aborting.")
                        break

            loop_i += 1
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n\nInterrupted. Stopping ...")
    finally:
        backend.stop(wait=True)
        teleop.disconnect()


if __name__ == "__main__":
    main()
