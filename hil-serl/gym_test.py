from lerobot.rl.gym_manipulator_general import GeneralRobotEnv
from lerobot.robots.rc10_follower import RC10FollowerConfig, RC10FollowerCut
from lerobot.teleoperators.space_mouse import SpaceMouseTeleopConfig, SpaceMouseTeleopCut
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.envs.configs import HILSerlRobotEnvConfig, HILSerlProcessorConfig, ObservationConfig, GripperConfig, ResetConfig
import numpy as np

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
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        # "side": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
        # "gripper": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS),

    },
    resolution=(224,224),
    limits = ((-0.5, 0.5), (-0.5, 0.5), (0.21, 0.5)),
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

gripper_config = GripperConfig(use_gripper=True,
                               gripper_penalty=0.01)

processor_config = HILSerlProcessorConfig(control_mode="ps4",
                                        observation=None,
                                        image_preprocessing=None,
                                        gripper=gripper_config,
                                        reset=None,
                                        inverse_kinematics=None,
                                        reward_classifier=None,
                                        max_gripper_pos=1)

hil_config = HILSerlRobotEnvConfig(task="hilserl_test",
                                   fps=30,
                                   max_parallel_tasks=1,
                                   robot=robot_config,
                                   teleop=teleop_config,
                                   processor=processor_config,
                                   name="Aaa")

FPS=30

robot = RC10FollowerCut(robot_config)

env = GeneralRobotEnv(robot=robot,
                      use_gripper=True,
                      display_cameras=False,
                      reset_pose=None,
                      reset_time_s=None)

print(env._obs_features)
print(env._image_keys)

print(env._obs_features - env._image_keys)
