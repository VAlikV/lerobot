from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kuka.act import record_kuka_act3 as recorder  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# All policy recording settings live in this file; argparse is not used.
ACTIVE_POLICY = "stage_2"
POLICIES_CONFIG_PATH = PROJECT_ROOT / "kuka/multi_policy/configs/policies.json"
ROBOT_CONFIG_PATH = PROJECT_ROOT / "kuka/act/configs/kuka_device_assemble.json"
START_POINTS_PATH = PROJECT_ROOT / "kuka/multi_policy/configs/kuka_policy_start_points.json"

RUNTIME_CONFIG_PATH = PROJECT_ROOT / "kuka/multi_policy/configs/active_recording_config.json"


class PolicyRecording:
    def __init__(self, config: dict[str, Any]) -> None:
        required_fields = {
            "name": str,
            "start_point": str,
            "dataset_repo_id": str,
            "task": str,
            "num_episodes": int,
            "episode_time_s": int,
            "action_recording_mode": str,
            "fps": int,
            "use_tts": bool,
            "show_force_vector": bool,
            "manual_reset_control": bool,
            "initial_gripper": str,
            "start_randomization": dict,
            "dagger": dict,
        }
        for field, expected_type in required_fields.items():
            if field not in config or not isinstance(config[field], expected_type):
                raise ValueError(f"{POLICIES_CONFIG_PATH}: field {field!r} must be {expected_type.__name__}")

        if config["action_recording_mode"] not in ("absolute", "relative_to_start"):
            raise ValueError("action_recording_mode must be 'absolute' or 'relative_to_start'")
        if config["initial_gripper"] not in ("manual", "open", "closed"):
            raise ValueError("initial_gripper must be 'manual', 'open', or 'closed'")
        if config["num_episodes"] <= 0 or config["episode_time_s"] <= 0 or config["fps"] <= 0:
            raise ValueError("num_episodes, episode_time_s and fps must be greater than zero")

        randomization = config["start_randomization"]
        for field in ("xy", "z", "yaw"):
            if field not in randomization or not isinstance(randomization[field], (int, float)):
                raise ValueError(f"start_randomization.{field} must be a number")

        dagger = config["dagger"]
        for field in ("model_dir", "dataset_repo_id"):
            if field not in dagger or (dagger[field] is not None and not isinstance(dagger[field], str)):
                raise ValueError(f"dagger.{field} must be a string or null")
        if not isinstance(dagger.get("device"), str):
            raise ValueError("dagger.device must be a string")
        if (dagger["model_dir"] is None) != (dagger["dataset_repo_id"] is None):
            raise ValueError("dagger.model_dir and dagger.dataset_repo_id must both be set or both be null")

        self.name = config["name"]
        self.start_point = config["start_point"]
        self.dataset_repo_id = config["dataset_repo_id"]
        self.task = config["task"]
        self.num_episodes = config["num_episodes"]
        self.episode_time_s = config["episode_time_s"]
        self.action_recording_mode = config["action_recording_mode"]
        self.fps = config["fps"]
        self.use_tts = config["use_tts"]
        self.show_force_vector = config["show_force_vector"]
        self.manual_reset_control = config["manual_reset_control"]
        self.initial_gripper = config["initial_gripper"]
        self.randomization_xy = float(randomization["xy"])
        self.randomization_z = float(randomization["z"])
        self.randomization_yaw = float(randomization["yaw"])
        self.dagger_model_dir = dagger["model_dir"]
        self.dagger_dataset_repo_id = dagger["dataset_repo_id"]
        self.dagger_device = dagger["device"]


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
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _find_start_pose(document: dict[str, Any], point_name: str) -> list[float]:
    points = document.get("points")
    if not isinstance(points, list):
        raise ValueError(f"Invalid start-points document: missing 'points' list in {START_POINTS_PATH}")

    matches = [point for point in points if isinstance(point, dict) and point.get("name") == point_name]
    if not matches:
        available_names = [point.get("name") for point in points if isinstance(point, dict)]
        raise KeyError(f"Start point {point_name!r} was not found. Available points: {available_names}")
    if len(matches) > 1:
        raise ValueError(f"Start point name {point_name!r} is duplicated in {START_POINTS_PATH}")

    pose = matches[0].get("pose")
    if not isinstance(pose, dict):
        raise ValueError(f"Start point {point_name!r} has no valid pose")
    missing_keys = [key for key in POSE_KEYS if key not in pose]
    if missing_keys:
        raise ValueError(f"Start point {point_name!r} is missing pose fields: {missing_keys}")
    return [float(pose[key]) for key in POSE_KEYS]


def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def _make_runtime_config(policy: PolicyRecording, start_pose: list[float]) -> dict[str, Any]:
    config = _read_json(ROBOT_CONFIG_PATH)
    start_pose = start_pose.copy()
    if policy.initial_gripper == "open":
        start_pose[6] = 1.0
    elif policy.initial_gripper == "closed":
        start_pose[6] = 0.0
    env_config = config["env"]
    robot_config = env_config["robot"]
    reset_config = env_config["processor"]["reset"]

    env_config["fps"] = policy.fps
    robot_config["reset_pose"] = start_pose
    reset_config["fixed_reset_joint_positions"] = start_pose
    reset_config["randomization_xy"] = policy.randomization_xy
    reset_config["randomization_z"] = policy.randomization_z
    reset_config["randomization_yaw"] = policy.randomization_yaw

    dataset_config = config["dataset"]
    dataset_config["repo_id"] = policy.dataset_repo_id
    dataset_config["task"] = policy.task
    dataset_config["num_episodes_to_record"] = policy.num_episodes
    dataset_config["action_recording_mode"] = policy.action_recording_mode
    return config


def _configure_recorder(policy: PolicyRecording) -> None:
    recorder.CONFIG_PATH = str(RUNTIME_CONFIG_PATH)
    recorder.REPO_ID = policy.dataset_repo_id
    recorder.TASK_DESCRIPTION = policy.task
    recorder.NUM_EPISODES = policy.num_episodes
    recorder.EPISODE_TIME_S = policy.episode_time_s
    recorder.FPS = policy.fps
    recorder.USE_TTS = policy.use_tts
    recorder.SHOW_FORCE_VECTOR = policy.show_force_vector
    recorder.MANUAL_RESET_CONTROL = policy.manual_reset_control
    recorder.RESET_GRIPPER_STATE = policy.initial_gripper
    recorder.POLICY_DIR = policy.dagger_model_dir
    recorder.POLICY_DATASET_REPO_ID = policy.dagger_dataset_repo_id
    recorder.POLICY_DEVICE = policy.dagger_device


def main() -> None:
    policies_config = _read_json(POLICIES_CONFIG_PATH)
    policy_definitions = policies_config.get("policies")
    if not isinstance(policy_definitions, dict):
        raise ValueError(f"{POLICIES_CONFIG_PATH}: 'policies' must be an object")
    if ACTIVE_POLICY not in policy_definitions:
        raise KeyError(
            f"Policy {ACTIVE_POLICY!r} is not defined. Available policies: {list(policy_definitions)}"
        )
    policy_config = policy_definitions[ACTIVE_POLICY]
    if not isinstance(policy_config, dict):
        raise ValueError(f"{POLICIES_CONFIG_PATH}: policy {ACTIVE_POLICY!r} must be an object")
    policy = PolicyRecording({**policy_config, "name": ACTIVE_POLICY})
    start_points = _read_json(START_POINTS_PATH)
    start_pose = _find_start_pose(start_points, policy.start_point)
    runtime_config = _make_runtime_config(policy, start_pose)
    _write_json_atomically(RUNTIME_CONFIG_PATH, runtime_config)
    _configure_recorder(policy)

    logger.info("Selected policy: %s", policy.name)
    logger.info("Start point: %s -> %s", policy.start_point, start_pose)
    logger.info("Dataset: %s", policy.dataset_repo_id)
    logger.info("Task: %s", policy.task)
    recorder.main()


if __name__ == "__main__":
    main()
