"""Simple KUKA iiwa teleoperation test using the Quest ROS 2 controller.

The Quest pose is relative to its calibration pose, while KUKA expects an
absolute Cartesian target. This script anchors Quest motion to the measured
KUKA pose at startup and composes rotations as quaternions instead of adding
Euler angles.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.teleoperators.quest3 import QuestRos2, QuestRos2Config
from lerobot.utils.robot_utils import precise_sleep

import threading

import multiprocess.resource_tracker

# Compatibility workaround for multiprocess 0.70.18 on CPython 3.12.0.
if not hasattr(threading.RLock(), "_recursion_count"):
    def _skip_incompatible_resource_tracker_finalizer(_resource_tracker: object) -> None:
        pass

    multiprocess.resource_tracker.ResourceTracker.__del__ = _skip_incompatible_resource_tracker_finalizer


# Edit these constants for the local setup. No argparse is used deliberately.
FPS = 30
URDF_PATH = Path(__file__).parents[2] / "robots" / "kuka_iiwa" / "iiwa2_gripper.urdf"
GRIPPER_PORT = "/dev/ttyACM0"

TF_TOPIC = "/tf"
JOY_TOPIC = "/quest/joystick"
TARGET_FRAME = "hand_right"

# "delta" moves only while the move axis is held. "absolute" continuously
# tracks the full pose relative to the calibration pose.
CONTROL_MODE = "delta"
POSITION_SCALE = 1.0
ROTATION_SCALE = 1.0

# Pose of the gripper relative to the tracked Quest controller. Quaternion
# order is [x, y, z, w]. Leave identity values when no offset is required.
# CONTROLLER_OFFSET_POSITION = (0.0, 0.0, 0.0)
# CONTROLLER_OFFSET_QUATERNION = (0.0, 0.0, 0.0, 1.0)

QUEST_GRIPPER_CLOSED = 1.0
KUKA_GRIPPER_OPEN = 1.0
KUKA_GRIPPER_CLOSED = -1.0


def pose_from_observation(observation: dict[str, Any]) -> tuple[np.ndarray, Rotation]:
    position = np.array(
        [observation["x.pos"], observation["y.pos"], observation["z.pos"]],
        dtype=np.float64,
    )
    rotation = Rotation.from_euler(
        "xyz",
        [observation["roll.pos"], observation["pitch.pos"], observation["yaw.pos"]],
        degrees=False,
    )
    return position, rotation


def gripper_to_kuka(quest_value: float) -> float:
    return KUKA_GRIPPER_CLOSED if quest_value >= QUEST_GRIPPER_CLOSED / 2 else KUKA_GRIPPER_OPEN


def integrate_delta_action(
    target_position: np.ndarray,
    target_rotation: Rotation,
    quest_action: dict[str, Any],
) -> tuple[np.ndarray, Rotation]:
    """Integrate a local Quest increment into the current absolute target."""
    position_delta = np.array(
        [-quest_action["x.delta"], -quest_action["y.delta"], quest_action["z.delta"]],
        dtype=np.float64,
    )
    rotation_delta = Rotation.from_euler(
        "xyz",
        [
            -quest_action["roll.delta"],
            -quest_action["pitch.delta"],
            quest_action["yaw.delta"],
        ],
        degrees=False,
    )
    return target_position + position_delta, target_rotation * rotation_delta


def apply_absolute_action(
    anchor_position: np.ndarray,
    anchor_rotation: Rotation,
    quest_action: dict[str, Any],
) -> tuple[np.ndarray, Rotation]:
    """Anchor the calibrated relative Quest pose to the initial KUKA pose."""
    relative_position = np.array(
        [quest_action["x.pos"], quest_action["y.pos"], quest_action["z.pos"]],
        dtype=np.float64,
    )
    relative_rotation = Rotation.from_euler(
        "xyz",
        [quest_action["roll.pos"], quest_action["pitch.pos"], quest_action["yaw.pos"]],
        degrees=False,
    )
    return anchor_position + relative_position, anchor_rotation * relative_rotation


def make_robot_action(
    position: np.ndarray,
    rotation: Rotation,
    gripper_position: float,
) -> dict[str, float]:
    roll, pitch, yaw = rotation.as_euler("xyz", degrees=False)
    return {
        "x.pos": float(position[0]),
        "y.pos": float(position[1]),
        "z.pos": float(position[2]),
        "roll.pos": float(roll),
        "pitch.pos": float(pitch),
        "yaw.pos": float(yaw),
        "gripper.pos": gripper_position,
    }


def main() -> None:
    if FPS <= 0:
        raise ValueError("FPS must be greater than zero")
    if CONTROL_MODE not in ("delta", "absolute"):
        raise ValueError("CONTROL_MODE must be 'delta' or 'absolute'")

    robot = KukaIiwa(
        KukaIiwaConfig(
            urdf_path=str(URDF_PATH),
            gripper_port=GRIPPER_PORT,
            cameras={},
        )
    )
    teleoperator = QuestRos2(
        QuestRos2Config(
            id="quest3_right",
            tf_topic=TF_TOPIC,
            joy_topic=JOY_TOPIC,
            target_frame=TARGET_FRAME,
            control_mode=CONTROL_MODE,
            position_scale=POSITION_SCALE,
            rotation_scale=ROTATION_SCALE,
            # controller_offset_position=CONTROLLER_OFFSET_POSITION,
            # controller_offset_quaternion=CONTROLLER_OFFSET_QUATERNION,
        )
    )

    try:
        print("Connecting to KUKA iiwa ...")
        robot.connect()
        anchor_position, anchor_rotation = pose_from_observation(robot.get_observation())
        target_position = anchor_position.copy()
        target_rotation = anchor_rotation

        print("Connecting to Quest ROS 2 ...")
        teleoperator.connect()
        print(
            f"Ready in {CONTROL_MODE!r} mode. "
            "Keep the robot's emergency stop available; press Ctrl+C to stop."
        )

        period_s = 1.0 / FPS
        while True:
            cycle_start = time.perf_counter()
            quest_action = teleoperator.get_action()

            if CONTROL_MODE == "delta":
                target_position, target_rotation = integrate_delta_action(
                    target_position,
                    target_rotation,
                    quest_action,
                )
            else:
                target_position, target_rotation = apply_absolute_action(
                    anchor_position,
                    anchor_rotation,
                    quest_action,
                )

            robot.send_action(
                make_robot_action(
                    target_position,
                    target_rotation,
                    gripper_to_kuka(float(quest_action["gripper.pos"])),
                )
            )
            precise_sleep(max(period_s - (time.perf_counter() - cycle_start), 0.0))
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        if teleoperator.is_connected:
            teleoperator.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
