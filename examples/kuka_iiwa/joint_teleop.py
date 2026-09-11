"""
KUKA Leader → KUKA IIWA Follower Teleoperation

The leader and follower are controlled directly in joint space.
Joint positions are sent from the leader to the follower as
absolute values.

Before running, configure only the values in the "USER CONFIGURATION" section below.
"""

import time

from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.teleoperators.kuka_leader import KukaLeader, KukaLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.processor import RobotAction, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import (
    GripperPositionToDiscrete,
)



# USER CONFIGURATION

FPS = 30

# Follower robot serial port
FOLLOWER_PORT = "..."

# Leader robot serial port
LEADER_PORT = "/dev/ttyACM0"

# Gripper thresholds in degrees
GRIPPER_THRESHOLD = 26.0



def main():

    follower_config = KukaIiwaConfig(
        port=FOLLOWER_PORT,
        id="my_kuka_iiwa",
        use_task_space=False,
        use_direct_joint_control=True,
        use_degrees=True,
    )

    leader_config = KukaLeaderConfig(
        port=LEADER_PORT,
        id="my_kuka_leader",
        use_degrees=True,
    )

    follower = KukaIiwa(follower_config)
    leader = KukaLeader(leader_config)

    gripper_pipeline = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            GripperPositionToDiscrete(
                threshold=GRIPPER_THRESHOLD,
                reverse=False
            )
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    follower.connect()
    leader.connect()

    try:
        print("Reading follower home position...")
        follower_obs = follower.get_observation()
        follower_obs["gripper.pos"] = 0.0

        input("\nPress Enter to sync leader to follower home position...")

        leader.send_goal_position(
            follower_obs,
            hold_s=3.0,
            tolerance_counts=50,
        )

        print("Starting joint-space teleoperation...")

        while True:
            t0 = time.perf_counter()

            leader_action = leader.get_action()

            leader_action = gripper_pipeline(leader_action)

            _ = follower.send_action(leader_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")

    finally:
        if leader.is_connected:
            leader.disconnect()

        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":
    main()
