"""
Leader arm returns joint angles but follower arm accepts EE position and orientation and gripper state (open/close). So, to work together action proccessor is needed. In `examples/kuka_leader_to_kuka_iiwa_teleop.py`:
- define `URDF_PATH` to follower's .urdf file
- define ports in `follower_config` and `leader_config`
- define `joint_names` for `follower_kinematics_solver` according to according to follower's urdf
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
)
from lerobot.teleoperators.kuka_leader import KukaLeader, KukaLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
URDF_PATH = "iiwa2_gripper.urdf"    # Set path
FOLLOWER_JOINT_NAMES = [    # Joint names according to follower's urdf
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]


def main():
    # Set port
    follower_config = KukaIiwaConfig(port="...", id="my_kuka_iiwa", use_degrees=False)
    # Set port
    leader_config = KukaLeaderConfig(port="/dev/ttyACM0", id="my_kuka_leader")

    follower = KukaIiwa(follower_config)
    leader = KukaLeader(leader_config)

    follower_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_base_link",      # Set target frame link
        joint_names=FOLLOWER_JOINT_NAMES,
    )

    leader_to_follower_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            LeaderJointDeltaToFollowerEE(
                kinematics=follower_kinematics_solver,
                leader_motor_names=leader.joint_names,
                euler_order="xyz",
                use_latched_reference=True,
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

    #init_rerun(session_name="kuka_leader_to_kuka_iiwa_teleop")

    print("Starting cartesian-space teleoperation...")

    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            follower_obs = follower.get_observation()

            # Get teleop observation
            leader_action = leader.get_action()

            # teleop joints -> robot EE action
            follower_action = leader_to_follower_ee((leader_action, follower_obs))

            # Send action to robot
            _ = follower.send_action(follower_action)

            #log_rerun_data(observation=follower_obs, action=follower_action)

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
