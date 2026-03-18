"""
RC10 HIL_SERl record dataset script
"""
import logging
import time

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.rc10_follower import RC10EnvConfig, RC10FollowerConfig, RC10FollowerCut, RC10RobotEnv
from lerobot.teleoperators.ps4_joystick import PS4JoystickTeleop, PS4JoystickTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD

logging.basicConfig(level=logging.INFO)

FPS = 30
NUM_EPISODES = 20
EPISODE_TIME_S = 30.0
RESET_TIME_S = 5.0
TASK_DESCRIPTION = "pick_and_place"
REPO_ID = "local/rc10_hilserl_demos"
IMAGE_SIZE = (128, 128)

# we will not need the crop params during recording, keep the full images,
# because if we want to change the crop params later then we might have to record the dataset again
IMAGE_CROP = {}

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
        "gripper": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS),

    },
    resolution=(128,128),
)

teleop_config = PS4JoystickTeleopConfig(
    id="my_teleop_ps4_joystick",
    max_speed=0.05,
    max_rot_speed=0.5,
    deadzone=0.05,
    alpha=0.3,
    poll_rate=100,
    x_init=0.095,
    y_init=0.35,
    z_init=0.23,
    roll_init=np.pi,
    pitch_init=0.0,
    yaw_init=0.0,
    delta_mode=True,  # normalized [-1, 1] deltas
)

env_config = RC10EnvConfig(
    ee_step_sizes={"x": 0.002, "y": 0.002, "z": 0.002, "yaw": 0.05},
    ee_bounds={
        "min": [-0.0771, 0.2554, 0.2296],   #obtain these using the rc10/rc10_find_ee_limits.py file
        "max": [0.2836, 0.6417, 0.4079],
    },
    fixed_roll=np.pi,
    fixed_pitch=0.0,
    reset_tcp= [0.095, 0.35, 0.23, np.pi, 0.0, 0.0],    # home pose
    reset_time_s=RESET_TIME_S,
    display_cameras=False,
    use_gripper=True,
)

def main():
    robot = RC10FollowerCut(robot_config)
    teleop = PS4JoystickTeleop(teleop_config)
    teleop.connect()

    # print(robot)

    env = RC10RobotEnv(robot=robot, env_config=env_config)

    max_steps = int(EPISODE_TIME_S * FPS)
    dt = 1.0/FPS

    features = {
        ACTION: {"dtype": "float32", "shape": (5,), "names": ["dx", "dy", "dz", "dyaw", "gripper"]},
        OBS_STATE: {"dtype": "float32", "shape": (5,), "names": None},
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
    }

    sample_obs, _ = env.reset()
    for cam_name, image in sample_obs["pixels"].items():
        resized = cv2.resize(image, IMAGE_SIZE)
        h, w, c = resized.shape
        features[f"{OBS_IMAGES}.{cam_name}"] = {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        REPO_ID,
        FPS,
        features=features,
        use_videos=True,
        image_writer_threads=4,
        )

    logging.info("=" * 50)
    logging.info(f"    Recording {NUM_EPISODES} episodes at {FPS} FPS")
    logging.info(f"    TASK: {TASK_DESCRIPTION}")
    logging.info()
    logging.info(" Hold R1 to control robot")
    logging.info(" Trinagle=Success, Circle=Fail, Square=REDO, Cross=GripperToggle")
    logging.info("=" * 50)

    episode_idx = 0
    while episode_idx < NUM_EPISODES:
        obs, _ = env.reset()
        logging.info(f"Episode {episode_idx + 1}/{NUM_EPISODES} - lets go!")

        for step in range(max_steps):
            step_start = time.perf_counter()

            # get teleop events
            events = teleop.get_teleop_events()
            is_intervention = events[TeleopEvents.IS_INTERVENTION]
            success = events[TeleopEvents.SUCCESS]
            terminate = events[TeleopEvents.TERMINATE_EPISODE]
            rerecord = events[TeleopEvents.RERECORD_EPISODE]

            if is_intervention:
                action = teleop.get_action()    # gets numpy [dx, dy, dz, dyaw, gripper]
            else:
                action = np.zeros(5, dtype=np.float32) # no movement
                action[4] = 1.0 # set gripper to open (1.0) because 0,0 or -1.0 will set it to close every time

            next_obs, _, _, _, _ = env.step(action)

            state = torch.from_numpy(obs["agent_pos"]).float()
            images = {}
            for cam_name, image in obs["pixels"].items():
                # if cam_name in IMAGE_CROP:
                #     top, left, height, width = IMAGE_CROP[cam_name]
                #     image = image[top:top + height, left:left + width]
                resized = cv2.resize(image, IMAGE_SIZE)
                images[f"{OBS_IMAGES}.{cam_name}"] = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0

            reward = 1.0 if success else 0.0
            done = success or terminate or (step >= max_steps -1)

            frame = {
                ACTION: torch.from_numpy(action).float(),
                OBS_STATE: state,
                REWARD: np.array([reward], dtype=np.float32),
                DONE: np.array([done], dtype=bool),
                **images,
                "task": TASK_DESCRIPTION,
            }

            dataset.add_frame(frame)
            obs = next_obs

            if rerecord:
                logging.info("Re-recording episode...")
                dataset.clear_episode_buffer()
                break

            if done:
                status = "SUCCESS" if success else ("TIMEOUT" if step >= max_steps -1 else "TERMINATED")
                logging.info(f"Episode {episode_idx + 1} - {status} ({step + 1} steps)")
                dataset.save_episode()
                episode_idx += 1
                break

            elapsed = time.perf_counter() - step_start
            time.sleep(max(dt - elapsed, 0.0))

    dataset.finalize()
    logging.info(f"Dataset saved at {REPO_ID}")

    env.close()
    teleop.disconnect()

if __name__ == "__main__":
    main()

