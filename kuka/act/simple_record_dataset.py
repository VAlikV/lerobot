from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.teleoperators.haptic import HapticTeleop, HapticTeleopConfig

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 15
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "KUKA cube"
VIDEO_CODEC = "auto"
ENCODER_THREADS = 1
ENCODER_QUEUE_MAXSIZE = FPS * 3

robot_cfg = KukaIiwaConfig(urdf_path="src/lerobot/robots/kuka_iiwa/iiwa.urdf",
                                gripper_port = "/dev/ttyACM1",
                                gripper_baudrate = 115200,
                                cameras={
                                    "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
                                    "side": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS),
                                    "gripper": OpenCVCameraConfig(index_or_path=5, width=640, height=480, fps=FPS),

                                },
                                reset_time_s=RESET_TIME_SEC,
                                )
    
robot = KukaIiwa(robot_cfg)

print("Connecting to KUKA iiwa ...")
robot.connect()
obs = robot.get_observation()

teleop_cfg = HapticTeleopConfig(ip="127.0.0.1",
                                port=8081,
                                delta_mode=False,
                                init_values=[obs["x.pos"],
                                                obs["y.pos"],
                                                obs["z.pos"],
                                                obs["roll.pos"], 
                                                obs["pitch.pos"], 
                                                obs["yaw.pos"]])


# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="local/ACT_KUKA_50eps",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    vcodec=VIDEO_CODEC,
    streaming_encoding=True,
    encoder_threads=ENCODER_THREADS,
    encoder_queue_maxsize=ENCODER_QUEUE_MAXSIZE,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

print("Connecting haptic ...")
teleop = HapticTeleop(teleop_cfg)
teleop.connect()

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

try:
    with VideoEncodingManager(dataset):
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

            if not events["stop_recording"]:
                log_say("Reset the environment")
                reset_action = robot.reset()
                teleop.reset(
                    init_values=[
                        reset_action["x.pos"],
                        reset_action["y.pos"],
                        reset_action["z.pos"],
                        reset_action["roll.pos"],
                        reset_action["pitch.pos"],
                        reset_action["yaw.pos"],
                    ],
                    gripper_pos=reset_action["gripper.pos"],
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_idx += 1
finally:
    # Clean up
    log_say("Stop recording")
    if robot.is_connected:
        robot.disconnect()
    if teleop.is_connected:
        teleop.disconnect()
