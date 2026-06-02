import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.utils.robot_utils import precise_sleep


CONFIG: dict[str, Any] = {
    "robot": {
        "urdf_path": "src/lerobot/robots/kuka_iiwa/iiwa.urdf",
        "gripper_port": "/dev/ttyUSB0",
        "gripper_baudrate": 115200,
        "cameras": {},
    },
    "teleop": {
        "use_gripper": True,
        "use_yaw": True,
        "invert_delta_x": False,
        "invert_delta_y": False,
        "invert_delta_z": False,
        "invert_delta_yaw": False,
    },
    "run": {
        "teleop_time_s": 120.0,
        "warmup_time_s": 5.0,
        "fps": 30,
    },
    "control": {
        "ee_step": [0.005, 0.005, 0.005],
        "yaw_step": 0.02,
    },
}


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).copy()


def _raw_observation(robot) -> np.ndarray:
    return _as_array(robot._controller.get_observation())


def _tcp_from_raw(raw_obs: np.ndarray) -> np.ndarray:
    position = raw_obs[7:10]
    rot_matrix = raw_obs[10:19].reshape(3, 3)
    rpy = Rotation.from_matrix(rot_matrix).as_euler("xyz", degrees=False)
    return np.concatenate([position, rpy])


def _read_state(
    robot,
    previous_joint_pos: np.ndarray | None,
    previous_time: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    now = time.perf_counter()
    raw_obs = _raw_observation(robot)

    joint_pos = raw_obs[:7].copy()
    tcp = _tcp_from_raw(raw_obs)

    controller = robot._controller
    if hasattr(controller, "get_current_joint_vel"):
        joint_vel = _as_array(controller.get_current_joint_vel())
    elif hasattr(controller, "get_joint_velocities"):
        joint_vel = _as_array(controller.get_joint_velocities())
    elif previous_joint_pos is not None and previous_time is not None and now > previous_time:
        joint_vel = (joint_pos - previous_joint_pos) / (now - previous_time)
    else:
        joint_vel = np.zeros_like(joint_pos)

    return joint_pos, joint_vel, tcp, now


def _send_gripper(robot, grip: int) -> None:
    if robot._gripper is None:
        return

    if grip == 0:
        robot._gripper.send(0)
    elif grip == 2:
        robot._gripper.send(2)


def main(config: dict[str, Any] | None = None) -> None:
    config = CONFIG if config is None else config

    from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
    from lerobot.teleoperators.gamepad import GamepadTeleop
    from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig

    robot_cfg = KukaIiwaConfig(**config["robot"])
    teleop_cfg = GamepadTeleopConfig(**config["teleop"])
    run_cfg = config["run"]
    control_cfg = config["control"]

    robot = KukaIiwa(robot_cfg)

    print("Connecting to KUKA iiwa ...")
    robot.connect()

    print("Connecting gamepad ...")
    teleop = GamepadTeleop(teleop_cfg)
    teleop.connect()

    ee_step = np.asarray(control_cfg["ee_step"], dtype=np.float32)
    yaw_step = float(control_cfg["yaw_step"])
    dt = 1.0 / float(run_cfg["fps"])

    previous_joint_pos = None
    previous_time = None
    joint_pos, _, tcp, previous_time = _read_state(robot, previous_joint_pos, previous_time)
    previous_joint_pos = joint_pos

    target_xyz = np.asarray(tcp[:3], dtype=np.float32)

    min_joint_pos = None
    max_joint_pos = None
    min_joint_vel = None
    max_joint_vel = None
    min_tcp = None
    max_tcp = None

    fixed_roll = fixed_pitch = fixed_yaw = None
    fixed_roll = tcp[3]
    fixed_pitch = tcp[4]
    fixed_yaw = tcp[5]
    initial_tcp: np.ndarray | None = None
    home_rotation = Rotation.from_euler("xyz", [tcp[3], tcp[4], tcp[5]])
    target_yaw_offset = 0.0
    min_yaw_offset = 0.0
    max_yaw_offset = 0.0

    start_t = time.perf_counter()
    warmup_done = False

    print()
    print("=" * 50)
    print(f"  WARMUP PHASE ({run_cfg['warmup_time_s']}s)")
    print("  Move freely and pre-pose the wrist. Data is NOT recorded yet.")
    print("=" * 50)
    print()

    try:
        while True:
            t0 = time.perf_counter()

            teleop_action = teleop.get_action()
            dx = float(teleop_action.get("delta_x", 0.0))
            dy = float(teleop_action.get("delta_y", 0.0))
            dz = float(teleop_action.get("delta_z", 0.0))
            dyaw = float(teleop_action.get("delta_yaw", 0.0)) if teleop_cfg.use_yaw else 0.0
            grip = int(teleop_action.get("gripper", 1))

            joint_pos, joint_vel, tcp, previous_time = _read_state(
                robot,
                previous_joint_pos,
                previous_time,
            )
            previous_joint_pos = joint_pos

            # print(dx,dy,dz,dy)

            # print(tcp)

            delta_xyz = np.array([dx, dy, dz], dtype=np.float32) * ee_step
            target_xyz = target_xyz + delta_xyz

            if fixed_roll is None:
                roll, pitch, yaw = (float(v) for v in tcp[3:6])
            elif teleop_cfg.use_yaw and home_rotation is not None:
                target_yaw_offset = float(
                    np.clip(target_yaw_offset + dyaw * yaw_step, -np.pi, np.pi)
                )
                min_yaw_offset = min(min_yaw_offset, target_yaw_offset)
                max_yaw_offset = max(max_yaw_offset, target_yaw_offset)
                target_rotation = home_rotation * Rotation.from_euler("z", target_yaw_offset)
                roll, pitch, yaw = (
                    float(v) for v in target_rotation.as_euler("xyz", degrees=False)
                )
            else:
                roll, pitch, yaw = fixed_roll, fixed_pitch, fixed_yaw

            robot.send_action(
                {
                    "x.pos": float(target_xyz[0]),
                    "y.pos": float(target_xyz[1]),
                    "z.pos": float(target_xyz[2]),
                    "roll.pos": roll,
                    "pitch.pos": pitch,
                    "yaw.pos": yaw,
                    "gripper.pos": float(grip),
                }
            )
            _send_gripper(robot, grip)

            elapsed = time.perf_counter() - start_t

            if elapsed >= float(run_cfg["warmup_time_s"]):
                if not warmup_done:
                    print()
                    print("=" * 50)
                    print("  RECORDING STARTED")
                    print("  Drive to ALL workspace extremes and exercise the gripper.")
                    print("  Press Ctrl+C to stop and see results.")
                    print("=" * 50)
                    print()

                    fixed_roll, fixed_pitch, fixed_yaw = (float(v) for v in tcp[3:6])
                    home_rotation = Rotation.from_euler(
                        "xyz",
                        [fixed_roll, fixed_pitch, fixed_yaw],
                        degrees=False,
                    )
                    initial_tcp = tcp.copy()
                    target_xyz = np.asarray(tcp[:3], dtype=np.float32)

                    min_joint_pos = joint_pos.copy()
                    max_joint_pos = joint_pos.copy()
                    min_joint_vel = joint_vel.copy()
                    max_joint_vel = joint_vel.copy()
                    min_tcp = tcp.copy()
                    max_tcp = tcp.copy()
                    warmup_done = True

                min_joint_pos = np.minimum(min_joint_pos, joint_pos)
                max_joint_pos = np.maximum(max_joint_pos, joint_pos)
                min_joint_vel = np.minimum(min_joint_vel, joint_vel)
                max_joint_vel = np.maximum(max_joint_vel, joint_vel)
                min_tcp = np.minimum(min_tcp, tcp)
                max_tcp = np.maximum(max_tcp, tcp)

                recording_time = elapsed - float(run_cfg["warmup_time_s"])
                remaining = float(run_cfg["teleop_time_s"]) - recording_time
                print(f"  Recording ... {remaining:.1f}s remaining", end="\r")

                if recording_time >= float(run_cfg["teleop_time_s"]):
                    print("\nTime limit reached.")
                    break

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n\nInterrupted. Stopping safely ...")

    finally:
        robot.disconnect()
        teleop.disconnect()

    if min_joint_pos is None or initial_tcp is None:
        print("No data recorded (exited during warmup).")
        return

    r = 4

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n# Joint Position Limits (radians) - 7 values [absolute]")
    print(f"min_joint_pos = {np.round(min_joint_pos, r).tolist()}")
    print(f"max_joint_pos = {np.round(max_joint_pos, r).tolist()}")

    print("\n# Joint Velocity Limits (rad/s) - 7 values [measured or finite-difference]")
    print(f"min_joint_vel = {np.round(min_joint_vel, r).tolist()}")
    print(f"max_joint_vel = {np.round(max_joint_vel, r).tolist()}")

    print("\n# TCP Limits [x, y, z, roll, pitch, yaw] [absolute base-frame]")
    print(f"min_tcp = {np.round(min_tcp, r).tolist()}")
    print(f"max_tcp = {np.round(max_tcp, r).tolist()}")

    print("\n# Wrist orientation captured at end of warmup")
    print(
        "fixed_roll, fixed_pitch, fixed_yaw = "
        f"{fixed_roll:.6f}, {fixed_pitch:.6f}, {fixed_yaw:.6f}"
    )

    print("\n# Home TCP pose [x, y, z, roll, pitch, yaw]")
    print(f"home_tcp_pose = {np.round(initial_tcp, r).tolist()}")

    print("\n# End-Effector XYZ Bounds")
    print(f"ee_bounds_min = {np.round(min_tcp[:3], r).tolist()}")
    print(f"ee_bounds_max = {np.round(max_tcp[:3], r).tolist()}")

    if teleop_cfg.use_yaw:
        print("\n# Yaw OFFSET bounds (radians, relative to warmup-end orientation)")
        print(f"yaw_offset_min = {round(min_yaw_offset, r)}")
        print(f"yaw_offset_max = {round(max_yaw_offset, r)}")

    gripper_min = 1.0
    gripper_max = 1.0
    obs_min = np.concatenate([min_tcp, [gripper_min]])
    obs_max = np.concatenate([max_tcp, [gripper_max]])

    print("\n# dataset_stats for KukaIiwa observation.state (7D)")
    print("# Order: [x, y, z, roll, pitch, yaw, gripper]")
    print(f'"min": {np.round(obs_min, r).tolist()}')
    print(f'"max": {np.round(obs_max, r).tolist()}')


if __name__ == "__main__":
    main()
