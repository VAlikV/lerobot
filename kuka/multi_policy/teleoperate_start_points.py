from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_robot_env
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POLICIES_CONFIG_PATH = PROJECT_ROOT / "kuka/multi_policy/configs/policies.json"
START_POINTS_PATH = PROJECT_ROOT / "kuka/multi_policy/configs/kuka_policy_start_points.json"
ROBOT_CONFIG_PATH = PROJECT_ROOT / "kuka/act/configs/kuka_device_assemble.json"

# The order is read from pipeline.order in policies.json.
# After the last point, SUCCESS returns to the first point when this is True.
CYCLE_START_POINTS = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSE_KEYS = (
    "x.pos",
    "y.pos",
    "z.pos",
    "roll.pos",
    "pitch.pos",
    "yaw.pos",
    "gripper.pos",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def _load_order_and_poses() -> tuple[list[str], dict[str, list[float]]]:
    policies_document = _read_json(POLICIES_CONFIG_PATH)
    pipeline = policies_document.get("pipeline")
    policies = policies_document.get("policies")
    if not isinstance(pipeline, dict) or not isinstance(policies, dict):
        raise ValueError(f"{POLICIES_CONFIG_PATH}: missing pipeline or policies object")

    order = pipeline.get("order")
    if not isinstance(order, list) or not order or not all(isinstance(name, str) for name in order):
        raise ValueError("pipeline.order must be a non-empty list of policy names")
    if len(order) != len(set(order)):
        raise ValueError("pipeline.order contains duplicate policy names")

    points_document = _read_json(START_POINTS_PATH)
    points = points_document.get("points")
    if not isinstance(points, list):
        raise ValueError(f"{START_POINTS_PATH}: missing points list")

    poses: dict[str, list[float]] = {}
    for policy_name in order:
        policy = policies.get(policy_name)
        if not isinstance(policy, dict):
            raise KeyError(f"Policy {policy_name!r} from pipeline.order is not defined")

        point_name = policy.get("start_point", policy_name)
        matches = [
            point
            for point in points
            if isinstance(point, dict) and point.get("name") == point_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Policy {policy_name!r}: expected one start point {point_name!r}, "
                f"found {len(matches)}"
            )
        pose = matches[0].get("pose")
        if not isinstance(pose, dict) or any(key not in pose for key in POSE_KEYS):
            raise ValueError(f"Start point {point_name!r} does not contain a complete pose")

        values = [float(pose[key]) for key in POSE_KEYS]
        initial_gripper = policy.get("initial_gripper", "manual")
        if initial_gripper == "open":
            values[6] = 1.0
        elif initial_gripper == "closed":
            values[6] = -1.0
        elif initial_gripper != "manual":
            raise ValueError(
                f"Policy {policy_name!r}: initial_gripper must be manual, open, or closed"
            )
        poses[policy_name] = values

    return order, poses


def _teleop_action(teleop_device, *, use_yaw: bool, use_gripper: bool) -> np.ndarray:
    action = teleop_device.get_action()
    values = [
        action.get("delta_x", 0.0),
        action.get("delta_y", 0.0),
        action.get("delta_z", 0.0),
    ]
    if use_yaw:
        values.append(action.get("delta_yaw", 0.0))
    if use_gripper:
        values.append(action.get("gripper", 1.0))
    return np.asarray(values, dtype=np.float32)


def _reset_to_point(env, policy_name: str, pose: list[float]) -> None:
    env.config.home_tcp = pose.copy()
    env.config.fixed_roll = float(pose[3])
    env.config.fixed_pitch = float(pose[4])
    env.config.fixed_yaw = float(pose[5])
    env.reset()
    print(f"[START POINT] {policy_name}: {pose}", flush=True)
    logger.info("Manual control at start point %s", policy_name)


def main() -> None:
    order, poses = _load_order_and_poses()
    policies_document = _read_json(POLICIES_CONFIG_PATH)
    pipeline = policies_document["pipeline"]

    with ROBOT_CONFIG_PATH.open(encoding="utf-8") as file:
        raw_robot_config = json.load(file)
    config = draccus.decode(GymManipulatorConfig, raw_robot_config)

    fps = int(pipeline.get("fps", config.env.fps or 30))
    reset_time_s = float(pipeline.get("reset_time_s", 5.0))
    if fps <= 0 or reset_time_s < 0:
        raise ValueError("pipeline fps must be positive and reset_time_s cannot be negative")
    if config.env.fps != fps:
        raise ValueError(f"Robot config fps={config.env.fps} does not match pipeline fps={fps}")

    env = None
    teleop_device = None
    try:
        env, teleop_device = make_robot_env(config.env)
        if teleop_device is None:
            raise RuntimeError("The robot config did not create a gamepad teleoperator")

        env.config.reset_time_s = reset_time_s
        gripper_config = config.env.processor.gripper
        ik_config = config.env.processor.inverse_kinematics
        use_gripper = bool(gripper_config.use_gripper) if gripper_config is not None else False
        use_yaw = bool(getattr(ik_config, "use_yaw", False)) if ik_config is not None else False
        dt_s = 1.0 / float(fps)

        active_index = 0
        _reset_to_point(env, order[active_index], poses[order[active_index]])
        logger.info("SUCCESS: next start point; FAILURE/RERECORD: stop.")

        success_was_pressed = False
        while True:
            step_start = time.perf_counter()
            action = _teleop_action(
                teleop_device,
                use_yaw=use_yaw,
                use_gripper=use_gripper,
            )
            env.step(action)
            events = teleop_device.get_teleop_events()
            success_pressed = bool(events.get(TeleopEvents.SUCCESS, False))

            if success_pressed and not success_was_pressed:
                next_index = active_index + 1
                if next_index >= len(order):
                    if not CYCLE_START_POINTS:
                        logger.info("Last start point reached. Stopping.")
                        break
                    next_index = 0

                active_index = next_index
                _reset_to_point(env, order[active_index], poses[order[active_index]])
                # Reset is blocking; sample the button again to prevent an held
                # press from immediately switching through another point.
                success_was_pressed = bool(
                    teleop_device.get_teleop_events().get(TeleopEvents.SUCCESS, False)
                )
                precise_sleep(max(dt_s - (time.perf_counter() - step_start), 0.0))
                continue

            success_was_pressed = success_pressed
            if events.get(TeleopEvents.TERMINATE_EPISODE, False):
                logger.info("Stop button pressed.")
                break

            precise_sleep(max(dt_s - (time.perf_counter() - step_start), 0.0))
    except KeyboardInterrupt:
        logger.info("Stopped by Ctrl+C.")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logger.exception("teleoperator disconnect failed")


if __name__ == "__main__":
    main()
