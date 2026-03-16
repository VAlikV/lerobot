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
import matplotlib.pyplot as plt

FPS = 30

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 10000


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    model_id = "outputs/robot_learning_tutorial/act/last"
    model = ACTPolicy.from_pretrained(model_id)

    dataset_id = "local/ACT_RC10_60eps_pcb"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    # preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=model_id, dataset_stats=dataset_metadata.stats)

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
            "side": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
            "gripper": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS)

        },
        resolution=(224,224),
        limits = ((-0.5, 0.5), (-0.5, 0.5), (0.21, 0.5)),
        action_pos_scale=1000,
        action_angle_scale=100
    )

    # Initialize the robot and teleoperator
    robot = RC10FollowerCut(robot_config)
    robot.connect()

    # plt.ion()  # интерактивный режим

    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((224,224,3)))

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)

            # print(obs_frame)

            # img.set_data(obs['observation.images.front'].cpu().numpy()[0].transpose((1,2,0)))
            # img.set_data(obs_frame['observation.images.front'].cpu().numpy()[0].transpose((1,2,0)))
            # plt.pause(0.01)  # ~30 FPS


            action = model.select_action(obs)
            action = postprocess(action)

            action = make_robot_action(action, dataset_metadata.features)

            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
