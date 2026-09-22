"""
KUKA Leader -> KUKA IIWA Follower Teleoperation + Dataset Recording

Before running, configure only the values in the "USER CONFIGURATION" section below.
"""

import time

from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig
from lerobot.teleoperators.kuka_leader import KukaLeader, KukaLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.processor import RobotAction, RobotProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import (
    GripperPositionToDiscrete,
    KukaJointBoundsAndSafety,
)

from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.keyboard_input import init_keyboard_listener


# USER CONFIGURATION

FPS = 30
FOLLOWER_URDF_PATH = "src/lerobot/robots/kuka_iiwa/iiwa2_gripper.urdf"
LEADER_PORT = "/dev/ttyACM0"
GRIPPER_PORT = None
# Gripper thresholds in degrees
GRIPPER_THRESHOLD = 26.0

REPO_ID = "local/kuka_test_1"
TASK_DESCRIPTION = "kuka_assemble"
RESUME = False
DATASET_ROOT = None
NUM_EPISODES = 2
EPISODE_TIME_S = 60
# Pause between episodes
RESET_TIME_S = 30
USE_TTS = True


@safe_stop_image_writer
def _record_episode(
    follower: KukaIiwa,
    leader: KukaLeader,
    pipeline: RobotProcessorPipeline,
    dataset: LeRobotDataset,
    events: dict,
    task: str,
    fps: int,
    duration_s: float,
) -> None:
    episode_start = time.perf_counter()
    while time.perf_counter() - episode_start < duration_s:
        t0 = time.perf_counter()

        obs = follower.get_observation()

        leader_action = leader.get_action()
        leader_action = pipeline(leader_action)

        sent_action = follower.send_action(leader_action)

        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
        dataset.add_frame({**observation_frame, **action_frame, "task": task})

        if events["exit_early"]:
            events["exit_early"] = False
            break

        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def _reset_phase(
    follower: KukaIiwa,
    leader: KukaLeader,
    pipeline: RobotProcessorPipeline,
    fps: int,
    duration_s: float,
) -> None:
    reset_start = time.perf_counter()
    while time.perf_counter() - reset_start < duration_s:
        t0 = time.perf_counter()
        leader_action = leader.get_action()
        leader_action = pipeline(leader_action)
        follower.send_action(leader_action)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def main():

    follower_config = KukaIiwaConfig(
        gripper_port=GRIPPER_PORT,
        id="my_kuka_iiwa",
        urdf_path=FOLLOWER_URDF_PATH,
        use_task_space=False,
        use_direct_joint_control=True,
    )

    leader_config = KukaLeaderConfig(
        port=LEADER_PORT,
        id="my_kuka_leader",
        use_degrees=True,
        alpha=0.2
    )

    follower = KukaIiwa(follower_config)
    leader = KukaLeader(leader_config)

    pipeline = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            KukaJointBoundsAndSafety(
                joint_offset_deg=2.0,
            ),
            GripperPositionToDiscrete(
                threshold=GRIPPER_THRESHOLD,
                reverse=False
            )
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    _, _, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=pipeline,
            initial_features=create_initial_features(action=follower.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
    )

    num_cameras = len(getattr(follower, "cameras", {}) or {})

    if RESUME:
        dataset = LeRobotDataset.resume(
            REPO_ID,
            root=DATASET_ROOT,
            image_writer_processes=0,
            image_writer_threads=4 * max(num_cameras, 1),
        )
        sanity_check_dataset_robot_compatibility(dataset, follower, FPS, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            REPO_ID,
            FPS,
            root=DATASET_ROOT,
            robot_type=follower.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * max(num_cameras, 1),
        )


    follower.connect()
    leader.connect()

    listener, events = init_keyboard_listener()

    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        # Sync
        print("Reading follower home position...")
        input("\nPress Enter to sync leader to follower home position...")

        follower_obs = follower.get_observation()
        follower_obs["gripper.pos"] = 0.0

        leader.send_goal_position(
            follower_obs,
            hold_s=2.0,
            tolerance_counts=50,
        )

        print("Starting record loop...")

        with VideoEncodingManager(dataset):
            episode_idx = 0
            while episode_idx < NUM_EPISODES and not events["stop_recording"]:
                episode_index = dataset.num_episodes

                if USE_TTS:
                    log_say(f"Recording episode {episode_index}", True)
                print(f"\n--- Recording episode {episode_index} (session {episode_idx + 1}/{NUM_EPISODES}) ---")

                # Main record loop
                _record_episode(
                    follower, leader, pipeline, dataset, events,
                    TASK_DESCRIPTION, FPS, EPISODE_TIME_S,
                )

                if events["rerecord_episode"]:
                    print("Re-recording episode (discarding last take)...")
                    if USE_TTS:
                        log_say("Re-record episode", True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Save episode
                dataset.save_episode()
                episode_idx += 1

                if events["stop_recording"]:
                    print("Stop requested, finishing early.")
                    break

                # Reset the environment if not stopping or re-recording
                if RESET_TIME_S > 0 and episode_idx < NUM_EPISODES:
                    if USE_TTS:
                        log_say("Reset the environment", True)
                    _reset_phase(follower, leader, pipeline, FPS, RESET_TIME_S)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")

    finally:
        if dataset:
            dataset.finalize()

        if leader.is_connected:
            leader.disconnect()

        if follower.is_connected:
            follower.disconnect()

        if listener is not None:
            listener.stop()

        print(f"Dataset finalized -> {REPO_ID}")


if __name__ == "__main__":
    main()