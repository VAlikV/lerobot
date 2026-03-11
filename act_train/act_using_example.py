import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.rc10_follower import RC10FollowerCut, RC10FollowerConfig
from lerobot.teleoperators.space_mouse import SpaceMouseTeleopConfig
from lerobot.teleoperators.space_mouse import SpaceMouseTeleopCut
import numpy as np

FPS = 30

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 1000


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    model_id = "outputs/robot_learning_tutorial/act"
    model = ACTPolicy.from_pretrained(model_id)

    dataset_id = "local/ACT_RC10_50eps"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

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
        limits = ((-0.5, 0.5), (-0.5, 0.5), (0.22, 0.5))
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
    )

    # Initialize the robot and teleoperator
    robot = RC10FollowerCut(robot_config)
    teleop = SpaceMouseTeleopCut(teleop_config)

    robot.connect()

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)

            action = make_robot_action(action, dataset_metadata.features)

            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
