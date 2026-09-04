"""
Gamepad teleop test for the process-based UR10e OSC controller.

The 500 Hz torque loop runs in `UR10OSCController` (its own real-time process); this
script only reads the gamepad and pushes absolute EE targets via `set_target()`. So
pygame polling / printing here cannot jitter the control loop — the whole point of
the rebuild.

Target shaping mirrors ur10_find_limits / the old test: latched target_xyz (deltas
applied to the previous command), orientation live during warmup then pinned (optional
yaw via R_home·R_z(offset)). R1 (deadman) gates whether gamepad deltas accumulate.

Run with the lerobot conda env, e.g.:
  python act_train/ur10_osc_gamepad.py --ip 192.168.0.100 --kp_pos 1000 --kp_rot 50
Set the payload on the TEACH PENDANT (Payload→Measure); this script trusts it
(set_payload=False) by default. Keep the e-stop in hand on the first run.
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ur10_osc_controller import UR10OSCController


def main():
    p = argparse.ArgumentParser(description="Gamepad test for the process-based UR10e OSC controller")
    p.add_argument("--ip", type=str, default="192.168.0.100")
    p.add_argument("--frequency", type=int, default=500)
    p.add_argument("--kp_pos", type=float, default=6000.0)
    p.add_argument("--kp_rot", type=float, default=100.0)
    p.add_argument("--damping_ratio_pos", type=float, default=1.0)
    p.add_argument("--damping_ratio_rot", type=float, default=1.0)
    p.add_argument("--error_delta_pos", type=float, default=0.05)
    p.add_argument("--error_delta_rot", type=float, default=0.3)
    p.add_argument("--tcp_offset", type=float, nargs=6, default=None)
    p.add_argument("--set_payload", action="store_true", default=False,
                   help="override pendant payload over RTDE (default: trust the pendant)")
    p.add_argument("--payload_mass", type=float, default=1.3)
    p.add_argument("--payload_cog", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--joints_init", type=float, nargs=6, default=None)
    # Optional joint-space DOB (default off; finalized controller is pure PD).
    p.add_argument("--dob", dest="use_dob", action="store_true", default=False,
                   help="enable the joint-space disturbance observer (shrinks PD steady-state error)")
    p.add_argument("--dob_g", type=float, default=6.0, help="DOB bandwidth; keep dob_g*dob_inertia < ~1")
    p.add_argument("--dob_inertia", type=float, default=0.08)
    p.add_argument("--dob_damping", type=float, default=0.3)
    p.add_argument("--dob_clip", type=float, default=40.0)
    p.add_argument("--no_soft_real_time", dest="soft_real_time", action="store_false", default=True,
                   help="disable SCHED_RR (use if you can't run with rt privileges)")
    # Teleop shaping.
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

    ctrl = UR10OSCController(
        args.ip, frequency=args.frequency,
        kp_pos=args.kp_pos, kp_rot=args.kp_rot,
        damping_ratio_pos=args.damping_ratio_pos, damping_ratio_rot=args.damping_ratio_rot,
        error_delta_pos=args.error_delta_pos, error_delta_rot=args.error_delta_rot,
        tcp_offset=args.tcp_offset, set_payload=args.set_payload,
        payload_mass=args.payload_mass, payload_cog=args.payload_cog,
        use_gripper=args.use_gripper, gripper_port=args.gripper_port,
        use_dob=args.use_dob, dob_g=args.dob_g, dob_inertia=args.dob_inertia,
        dob_damping=args.dob_damping, dob_clip=args.dob_clip,
        joints_init=args.joints_init, soft_real_time=args.soft_real_time, verbose=False,
    )

    print(f"Starting OSC controller process (UR10e @ {args.ip}, {args.frequency} Hz) ...")
    ctrl.start(wait=True)
    print("Controller ready.")

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

    # Stick-bias calibration (hands off) — same rationale as the old script: latched
    # accumulation makes any resting bias creep the target.
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

    target_xyz = np.array(ctrl.get_current_tcp()[:3], dtype=np.float32)
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
            if not ctrl.is_alive():
                print("\n[main] controller process died — aborting."); break

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

            tcp = ctrl.get_current_tcp()
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

            ctrl.set_target([float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]), rx, ry, rz],
                            close_gripper=gripper_close)

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
                    extra = ""
                    if args.use_dob:
                        extra = f"   d_hat = {ctrl.get_state()['dhat']:5.1f} Nm"
                    print(f"  [{gate}]  pos_err = {err_mm:6.2f} mm  (max {max_err_mm:6.2f}){extra}", end="\r")

            loop_i += 1
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n\nInterrupted. Stopping ...")
    finally:
        ctrl.stop(wait=True)
        teleop.disconnect()


if __name__ == "__main__":
    main()
