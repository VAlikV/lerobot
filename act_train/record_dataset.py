from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.rc10_follower import RC10FollowerCut, RC10FollowerConfig
from lerobot.teleoperators.space_mouse import SpaceMouseTeleopConfig
from lerobot.teleoperators.space_mouse import SpaceMouseTeleopCut
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

import numpy as np

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 15
TASK_DESCRIPTION = "My task description"

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
        "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
        "side": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS)

    },
    resolution=(224,224),
    limits = ((-0.5, 0.5), (-0.5, 0.5), (0.22, 0.5)),
    action_pos_scale=1000,
    action_angle_scale=100
)

teleop_config = SpaceMouseTeleopConfig(
    id="my_teleop_space_mouse",
    max_speed=0.15,
    max_rot_speed=1.2,
    deadzone=200.0,
    alpha=0.3,
    poll_rate=100,
    device_num=0,
    x_init=0.5,
    y_init=0.5,
    z_init=0.5,
    roll_init=np.pi,
    pitch_init=0.0,
    yaw_init=0.0,
    action_pos_scale=1000,
    action_angle_scale=100
)

# Initialize the robot and teleoperator
robot = RC10FollowerCut(robot_config)
teleop = SpaceMouseTeleopCut(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="local/ACT_RC10_50eps",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    video_backend="libx264",
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
# dataset
# dataset.save_episode
dataset.finalize()