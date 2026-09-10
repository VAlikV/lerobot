import time

from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.teleoperators.kuka_leader import KukaLeader, KukaLeaderConfig
from lerobot.utils.robot_utils import precise_sleep


FPS = 30
JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]


def main():
    # Follower: direct joint-space control
    follower_config = KukaIiwaConfig(
        port="...",                 
        id="my_kuka_iiwa",

        use_task_space=False,
        use_direct_joint_control=True,

        # Joint commands/observations are absolute degrees
        use_degrees=True
    )

    # Leader
    leader_config = KukaLeaderConfig(
        port="/dev/ttyACM0",
        id="my_kuka_leader",

        use_degrees=True,
    )

    follower = KukaIiwa(follower_config)
    leader = KukaLeader(leader_config)

    follower.connect()
    leader.connect()


    try:
        print("Reading follower home position...")
        follower_obs = follower.get_observation()

        print("Syncing leader to follower home position...")
        leader.send_goal_position(follower_obs, hold_s=3.0, tolerance_counts=50)

        print("Leader synced to follower home position.")
        print("Starting joint-space teleoperation...")

        while True:
            t0 = time.perf_counter()

            # Read leader joint positions
            leader_action = leader.get_action()

            # Directly send joint positions to follower
            follower_action = {
                joint: leader_action[f"{joint}.pos"] for joint in JOINT_NAMES
            }

            # If leader provides a gripper command, forward it too.
            if "gripper.pos" in leader_action:
                follower_action["gripper.pos"] = leader_action["gripper.pos"]

            # Send absolute joint positions to KUKA iiwa
            _ = follower.send_action(follower_action)

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