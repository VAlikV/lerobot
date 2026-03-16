import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.robots.so_follower.config_so100_follower import SO100FollowerConfig
# from lerobot.robots.so_follower.so100_follower import SO100Follower
from lerobot.robots.rc10_follower import RC10FollowerCut, RC10FollowerConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

episode_idx = 0

# Create robot configuration
robot_config = RC10FollowerConfig(
    id="my_rc10_follower",
    ip="10.10.10.10",
    rate_hz=100,
    velocity=1.0,
    acceleration=1.0,
    threshold_position=0.001,
    threshold_angle=1.0,
    gripper_port="/dev/ttyUSB0",
    gripper_baudrate=115200,
    cameras={
        "front": OpenCVCameraConfig(index_or_path=3, width=640, height=480, fps=30),
        "side": OpenCVCameraConfig(index_or_path=5, width=640, height=480, fps=30)

    },
    resolution=(224,224),
    limits = ((-0.5, 0.5), (-0.5, 0.5), (0.22, 0.5)),
    action_pos_scale=1000,
    action_angle_scale=100
)

# Initialize the robot and teleoperator
robot = RC10FollowerCut(robot_config)
robot.connect()

dataset = LeRobotDataset("local/ACT_RC10_50eps3", episodes=[episode_idx])
actions = dataset.hf_dataset.select_columns("action")

log_say(f"Replaying episode {episode_idx}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    robot.send_action(action)

    precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()