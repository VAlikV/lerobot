from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_robot_env
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# User settings. This script intentionally does not use argparse.
CONFIG_PATH = Path("kuka/act/configs/kuka_device_assemble.json")
OUTPUT_PATH = Path("kuka/multi_policy/configs/kuka_policy_start_points.json")

# Names are assigned in this order. Add one name for each atomic policy.
# If the list is exhausted, names such as start_004 are generated automatically.
START_POINT_NAMES = [
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "stage_5",
    "stage_6",
    "stage_7",
    "stage_8",
    "stage_9",
]

# Triangle records a point. Circle/Cross (FAILURE) stops the script.
RECORD_EVENT = TeleopEvents.SUCCESS
STOP_EVENT = TeleopEvents.TERMINATE_EPISODE


POSE_KEYS = (
    "x.pos",
    "y.pos",
    "z.pos",
    "roll.pos",
    "pitch.pos",
    "yaw.pos",
    "gripper.pos",
)


def _load_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "robot_type": "kuka_iiwa",
            "config_path": str(CONFIG_PATH),
            "points": [],
        }

    with path.open(encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document, dict) or not isinstance(document.get("points"), list):
        raise ValueError(f"Invalid start-points file: {path}")
    return document


def _save_document(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace the JSON file so interruption cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def _next_point_name(point_index: int) -> str:
    if point_index < len(START_POINT_NAMES):
        return START_POINT_NAMES[point_index]
    return f"start_{point_index + 1:03d}"


def _read_pose(env) -> dict[str, float]:
    raw_pose = env.robot._get_pose_observation()
    missing_keys = [key for key in POSE_KEYS if key not in raw_pose]
    if missing_keys:
        raise KeyError(f"KUKA pose does not contain: {missing_keys}")
    return {key: float(raw_pose[key]) for key in POSE_KEYS}


def _teleop_action_vector(teleop_device, *, use_yaw: bool, use_gripper: bool) -> np.ndarray:
    action = teleop_device.get_action()
    values = [action.get("delta_x", 0.0), action.get("delta_y", 0.0), action.get("delta_z", 0.0)]
    if use_yaw:
        values.append(action.get("delta_yaw", 0.0))
    if use_gripper:
        values.append(action.get("gripper", 1.0))
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        raw_config = json.load(file)
    config = draccus.decode(GymManipulatorConfig, raw_config)

    env, teleop_device = make_robot_env(config.env)
    if teleop_device is None:
        env.close()
        raise RuntimeError("The selected config did not create a teleoperator.")

    ik_config = config.env.processor.inverse_kinematics
    gripper_config = config.env.processor.gripper
    use_yaw = bool(getattr(ik_config, "use_yaw", False)) if ik_config is not None else False
    use_gripper = bool(gripper_config.use_gripper) if gripper_config is not None else False
    fps = int(config.env.fps or 30)
    period_s = 1.0 / fps

    document = _load_document(OUTPUT_PATH)
    point_index = len(document["points"])

    try:
        env.reset()
        logger.info("Manual start-point recording is ready at %d Hz.", fps)
        logger.info("Triangle: save point; Circle/Cross: stop; sticks/triggers: move robot.")
        logger.info("Existing points: %d. Next name: %s", point_index, _next_point_name(point_index))

        record_button_was_down = False
        while True:
            loop_started_at = time.perf_counter()
            action = _teleop_action_vector(
                teleop_device,
                use_yaw=use_yaw,
                use_gripper=use_gripper,
            )
            env.step(action)

            events = teleop_device.get_teleop_events()
            record_button_is_down = bool(events.get(RECORD_EVENT, False))
            if record_button_is_down and not record_button_was_down:
                name = _next_point_name(point_index)
                point = {
                    "name": name,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "pose": _read_pose(env),
                }
                document["points"].append(point)
                _save_document(OUTPUT_PATH, document)
                point_index += 1
                logger.info("Saved %s to %s: %s", name, OUTPUT_PATH, point["pose"])
                logger.info("Next name: %s", _next_point_name(point_index))
            record_button_was_down = record_button_is_down

            if events.get(STOP_EVENT, False):
                logger.info("Stop button pressed.")
                break

            precise_sleep(max(period_s - (time.perf_counter() - loop_started_at), 0.0))
    except KeyboardInterrupt:
        logger.info("Stopped by Ctrl+C.")
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")
        try:
            teleop_device.disconnect()
        except Exception:
            logger.exception("teleoperator disconnect failed")


if __name__ == "__main__":
    main()
