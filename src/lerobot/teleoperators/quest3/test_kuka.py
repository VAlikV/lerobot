"""Minimal Cartesian teleoperation of a KUKA iiwa from a Quest controller.

Quest pose differences are integrated into an absolute target pose because
``KukaIiwa.send_action`` accepts absolute Cartesian actions.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.teleoperators.quest3 import QuestRos2, QuestRos2Config
from lerobot.utils.robot_utils import precise_sleep


FPS = 30
URDF_PATH = Path(__file__).parents[2] / "robots" / "kuka_iiwa" / "iiwa.urdf"
GRIPPER_PORT = "/dev/ttyACM0"

TF_TOPIC = "/tf"
JOY_TOPIC = "/quest/joystick"
TARGET_FRAME = "hand_right"
POSITION_SCALE = 0.3
ROTATION_SCALE = 0.3


def integrate_delta_action(
    target_position: np.ndarray,
    target_rotation: Rotation,
    quest_action: dict[str, float],
) -> tuple[np.ndarray, Rotation, float]:
    """Integrate one Quest delta into an absolute KUKA target.

    The Quest orientation delta is expressed relative to its preceding pose,
    so it is composed with the current target rotation instead of adding Euler
    angles component by component.
    """
    position_delta = np.array(
        [
            quest_action["x.delta"],
            quest_action["y.delta"],
            quest_action["z.delta"],
        ],
        dtype=np.float64,
    )
    rotation_delta = Rotation.from_euler(
        "xyz",
        [
            quest_action["roll.delta"],
            quest_action["pitch.delta"],
            quest_action["yaw.delta"],
        ],
        degrees=False,
    )

    target_position = target_position + position_delta
    target_rotation = target_rotation * rotation_delta

    # Quest: 1 means closed, 0 means open. KUKA: -1 closed, +1 open.
    gripper_position = -1.0 if quest_action["gripper.pos"] >= 0.5 else 1.0
    return target_position, target_rotation, gripper_position


def main() -> None:
    if FPS <= 0:
        raise ValueError("FPS must be greater than zero")

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
            position_scale=POSITION_SCALE,
            rotation_scale=ROTATION_SCALE,
        )
    )

    try:
        print("Connecting to KUKA iiwa ...")
        robot.connect()

        initial_pose = robot.get_observation()
        target_position = np.array(
            [initial_pose["x.pos"], initial_pose["y.pos"], initial_pose["z.pos"]],
            dtype=np.float64,
        )
        target_rotation = Rotation.from_euler(
            "xyz",
            [initial_pose["roll.pos"], initial_pose["pitch.pos"], initial_pose["yaw.pos"]],
            degrees=False,
        )

        print("Connecting to Quest ROS 2 ...")
        teleoperator.connect()
        print("Ready. Hold the move button to move; press Ctrl+C to stop.")

        period_s = 1.0 / FPS
        while True:
            cycle_start = time.perf_counter()
            quest_action = teleoperator.get_action()
            target_position, target_rotation, gripper_position = integrate_delta_action(
                target_position,
                target_rotation,
                quest_action,
            )
            roll, pitch, yaw = target_rotation.as_euler("xyz", degrees=False)

            robot.send_action(
                {
                    "x.pos": float(target_position[0]),
                    "y.pos": float(target_position[1]),
                    "z.pos": float(target_position[2]),
                    "roll.pos": float(roll),
                    "pitch.pos": float(pitch),
                    "yaw.pos": float(yaw),
                    "gripper.pos": gripper_position,
                }
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
