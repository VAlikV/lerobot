"""Find the UR10e workspace EE bounds (+ yaw range) for the standalone ACT scripts.

Jog the arm with the gamepad (OSC controller backend) to every extreme of the task
workspace; this tracks the min/max of the MEASURED TCP xyz and the commanded yaw
offset, then prints a ready-to-paste snippet:

    EE_BOUNDS_MIN = (xmin, ymin, zmin)
    EE_BOUNDS_MAX = (xmax, ymax, zmax)
    YAW_MIN, YAW_MAX = (..., ...)

Paste those over the EE_BOUNDS_* / YAW_* constants in record_ur10_follower.py and
eval_ur10_follower.py (they must match). A small margin is also printed (slightly
shrunk) so the clip stays comfortably reachable.

Controls: HOLD R1 to drive (deadman). Ctrl+C to finish and print results.
Orientation is pinned to the live wrist at the end of warmup, so pre-pose the wrist
during warmup, then drive to all corners + rotate the wrist to its yaw extremes.

Run (lerobot conda env; pendant payload set; e-stop in hand):
    python ur10_standalone_act/find_limits.py --ip 192.168.0.100
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ur10_osc_controller import UR10OSCController


def main() -> None:
    p = argparse.ArgumentParser(description="Find UR10e EE/yaw bounds via gamepad jog")
    p.add_argument("--ip", type=str, default="192.168.0.100")
    p.add_argument("--frequency", type=int, default=500)
    p.add_argument("--kp_pos", type=float, default=5000.0)
    p.add_argument("--kp_rot", type=float, default=100.0)
    p.add_argument("--set_payload", action="store_true", default=False)
    p.add_argument("--payload_mass", type=float, default=1.3)
    p.add_argument("--no_soft_real_time", dest="soft_real_time", action="store_false", default=True)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--warmup_time_s", type=float, default=5.0)
    p.add_argument("--margin_mm", type=float, default=5.0, help="shrink printed bounds by this (mm)")
    p.add_argument("--ee_step_x", type=float, default=0.001)
    p.add_argument("--ee_step_y", type=float, default=0.001)
    p.add_argument("--ee_step_z", type=float, default=0.001)
    p.add_argument("--yaw_step", type=float, default=0.006)
    p.add_argument("--invert_delta_x", action="store_true", default=True)
    p.add_argument("--invert_delta_y", action="store_true", default=True)
    p.add_argument("--invert_delta_z", action="store_true", default=False)
    p.add_argument("--invert_delta_yaw", action="store_true", default=False)
    p.add_argument("--stick_deadzone", type=float, default=0.05)
    p.add_argument("--stick_cal_s", type=float, default=1.5)
    args = p.parse_args()

    from lerobot.teleoperators.gamepad import GamepadTeleop
    from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
    from lerobot.utils.robot_utils import precise_sleep

    ctrl = UR10OSCController(
        args.ip, frequency=args.frequency, kp_pos=args.kp_pos, kp_rot=args.kp_rot,
        set_payload=args.set_payload, payload_mass=args.payload_mass,
        use_gripper=True, soft_real_time=args.soft_real_time,
    )
    print(f"Starting OSC controller (UR10e @ {args.ip}) ...")
    ctrl.start(wait=True)
    teleop = GamepadTeleop(GamepadTeleopConfig(
        use_gripper=True, use_yaw=True,
        invert_delta_x=args.invert_delta_x, invert_delta_y=args.invert_delta_y,
        invert_delta_z=args.invert_delta_z, invert_delta_yaw=args.invert_delta_yaw,
    ))
    teleop.connect()

    ee_step = np.array([args.ee_step_x, args.ee_step_y, args.ee_step_z], dtype=np.float32)
    dt = 1.0 / args.fps
    dz = args.stick_deadzone

    def _read():
        a = teleop.get_action()
        return a, np.array([
            float(a.get("delta_x", 0.0)), float(a.get("delta_y", 0.0)),
            float(a.get("delta_z", 0.0)), float(a.get("delta_yaw", 0.0))], dtype=np.float32)

    # Stick-bias cal.
    print(f"\n[stick-cal] release sticks for {args.stick_cal_s:.1f}s ...")
    n_cal = max(1, int(args.stick_cal_s * args.fps))
    bias = np.zeros(4, dtype=np.float32)
    for _ in range(n_cal):
        _, d = _read(); bias += d; precise_sleep(dt)
    bias /= n_cal

    def _dz(v):
        return 0.0 if abs(v) < dz else float(v)

    target_xyz = np.array(ctrl.get_current_tcp()[:3], dtype=np.float32)
    fixed_rot = None
    home_rot = None
    target_yaw = 0.0
    tcp_min = tcp_max = None
    yaw_min = yaw_max = 0.0
    start_t = time.perf_counter()
    warmup_done = False

    print(f"\nWARMUP ({args.warmup_time_s}s): pre-pose the wrist. Then HOLD R1 and drive to "
          "ALL corners + rotate the wrist to its yaw extremes. Ctrl+C to finish.")
    try:
        while True:
            t0 = time.perf_counter()
            if not ctrl.is_alive():
                print("\n[main] controller died — aborting."); break
            a, d = _read()
            enabled = bool(teleop.gamepad.should_intervene())
            if enabled:
                target_xyz = target_xyz + np.array(
                    [_dz(d[0] - bias[0]), _dz(d[1] - bias[1]), _dz(d[2] - bias[2])],
                    dtype=np.float32) * ee_step
                if warmup_done:
                    target_yaw = float(np.clip(target_yaw + _dz(d[3] - bias[3]) * args.yaw_step,
                                               -np.pi, np.pi))

            tcp = ctrl.get_current_tcp()
            if not warmup_done:
                rx, ry, rz = float(tcp[3]), float(tcp[4]), float(tcp[5])  # follow live wrist
            else:
                R_t = home_rot * Rot.from_euler("z", target_yaw)
                rx, ry, rz = (float(v) for v in R_t.as_rotvec())
            ctrl.set_target([float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]), rx, ry, rz])

            elapsed = time.perf_counter() - start_t
            if elapsed < args.warmup_time_s:
                print(f"  warmup ... {args.warmup_time_s - elapsed:4.1f}s ", end="\r")
            else:
                if not warmup_done:
                    fixed_rot = tcp[3:6].copy()
                    home_rot = Rot.from_rotvec(fixed_rot)
                    target_xyz = np.array(tcp[:3], dtype=np.float32)
                    target_yaw = 0.0
                    tcp_min = tcp[:3].copy()
                    tcp_max = tcp[:3].copy()
                    yaw_min = yaw_max = 0.0
                    warmup_done = True
                    print("\nRECORDING extremes — drive everywhere. Ctrl+C to finish.")
                m = tcp[:3]
                tcp_min = np.minimum(tcp_min, m)
                tcp_max = np.maximum(tcp_max, m)
                yaw_min = min(yaw_min, target_yaw)
                yaw_max = max(yaw_max, target_yaw)
                box = (tcp_max - tcp_min) * 1000
                print(f"  x[{tcp_min[0]:+.3f},{tcp_max[0]:+.3f}] y[{tcp_min[1]:+.3f},{tcp_max[1]:+.3f}] "
                      f"z[{tcp_min[2]:+.3f},{tcp_max[2]:+.3f}] box={box.round(0)}mm "
                      f"yaw[{yaw_min:+.3f},{yaw_max:+.3f}]   ", end="\r")
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n\nFinished.")
    finally:
        ctrl.stop(wait=True)
        teleop.disconnect()

    if tcp_min is None:
        print("No data (exited during warmup).")
        return

    mm = args.margin_mm / 1000.0
    bmin = (tcp_min + mm).round(4)
    bmax = (tcp_max - mm).round(4)
    print("\n" + "=" * 64)
    print("RESULTS — paste into record_ur10_follower.py AND eval_ur10_follower.py")
    print("=" * 64)
    print("# raw measured extremes:")
    print(f"#   min {tcp_min.round(4).tolist()}  max {tcp_max.round(4).tolist()}  "
          f"yaw [{round(yaw_min,4)}, {round(yaw_max,4)}]")
    print(f"# with {args.margin_mm:.0f}mm safety margin:")
    print(f"EE_BOUNDS_MIN = ({bmin[0]}, {bmin[1]}, {bmin[2]})")
    print(f"EE_BOUNDS_MAX = ({bmax[0]}, {bmax[1]}, {bmax[2]})")
    print(f"YAW_MIN, YAW_MAX = {round(yaw_min, 4)}, {round(yaw_max, 4)}")


if __name__ == "__main__":
    main()
