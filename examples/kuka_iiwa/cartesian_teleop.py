"""
KUKA Leader → KUKA IIWA Follower Teleoperation

The leader arm returns joint angles, while the follower accepts
end-effector position, orientation, and gripper state.

Before running, configure the values in the "USER CONFIGURATION" section below.
"""

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import (
    KukaEEBoundsAndSafety,
    LeaderJointDeltaToFollowerEE,
    GripperPositionToDiscrete,
)
from lerobot.teleoperators.kuka_leader import KukaLeader, KukaLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data



# USER CONFIGURATION

FPS = 30

# Follower robot URDF
URDF_PATH = "iiwa2_gripper.urdf"

# Follower joint names — must match the joint names in the URDF
FOLLOWER_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

# Follower robot serial port
FOLLOWER_PORT = "..."

# Leader robot serial port
LEADER_PORT = "/dev/ttyACM0"

# End-effector frame name from the follower URDF
FOLLOWER_EE_FRAME = "gripper_base_link"

# Gripper thresholds in radians
GRIPPER_THRESHOLD = 0.45



def main():

    follower_config = KukaIiwaConfig(
        port=FOLLOWER_PORT,
        id="my_kuka_iiwa",
        use_degrees=False,
    )

    leader_config = KukaLeaderConfig(
        port=LEADER_PORT,
        id="my_kuka_leader",
    )

    follower = KukaIiwa(follower_config)
    leader = KukaLeader(leader_config)

    follower_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=FOLLOWER_EE_FRAME,
        joint_names=FOLLOWER_JOINT_NAMES,
    )

    leader_to_follower_ee = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            LeaderJointDeltaToFollowerEE(
                kinematics=follower_kinematics_solver,
                leader_motor_names=leader.joint_names[:-1],
                euler_order="xyz",
                use_latched_reference=True,
            ),
            GripperPositionToDiscrete(
                threshold=GRIPPER_THRESHOLD,
            ),
            KukaEEBoundsAndSafety(
                end_effector_bounds={"min": None, "max": None},
                max_ee_step_m=0.05,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    follower.connect()
    leader.connect()

    print("Starting cartesian-space teleoperation...")

    try:
        while True:
            t0 = time.perf_counter()

            follower_obs = follower.get_observation()
            leader_action = leader.get_action()

            follower_action = leader_to_follower_ee(
                (leader_action, follower_obs)
            )

            _ = follower.send_action(follower_action)

            precise_sleep(
                max(1.0 / FPS - (time.perf_counter() - t0), 0.0)
            )

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")

    finally:
        if leader.is_connected:
            leader.disconnect()

        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":
    main()
