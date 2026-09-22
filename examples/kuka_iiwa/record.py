"""
KUKA Leader -> KUKA IIWA Follower Teleoperation + Dataset Recording

Keyboard Controls During Recording:
--> / n - Prematurely terminate the current episode / reset; proceed to the next. 
<-- / r - Cancel the current episode; re-record.
esc / q - Immediately stop the session.
"""

import json
import time
import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig
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

from record_config import (
    RecordingConfig,
    PipelineConfig,
    DatasetConfig,
)

CONFIG_PATH = "lerobot/examples/kuka_iiwa/configs/record_config.json"

# Optionally can be switch to record_loop from lerobot.scripts.lerobot_record
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


def load_config(path: str) -> RecordingConfig:
    with open(path, "r") as f:
        raw_cfg = json.load(f)

    return draccus.decode(
        RecordingConfig,
        raw_cfg,
    )


def main():

    cfg = load_config(CONFIG_PATH)

    fps = cfg.fps

    follower = KukaIiwa(cfg.robot)
    leader = KukaLeader(cfg.leader)

    pipeline = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            KukaJointBoundsAndSafety(
                joint_offset_deg=cfg.pipeline.joint_offset_deg,
            ),
            GripperPositionToDiscrete(
                threshold=cfg.pipeline.gripper_threshold,
                reverse=cfg.pipeline.gripper_reverse,
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

    if cfg.dataset.resume:
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            image_writer_processes=0,
            image_writer_threads=4 * max(num_cameras, 1),
        )
        sanity_check_dataset_robot_compatibility(dataset, follower, fps, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            fps,
            root=cfg.dataset.root,
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
            while episode_idx < cfg.dataset.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes

                if cfg.dataset.use_tts:
                    log_say(f"Recording episode {episode_index}", True)
                print(f"\n--- Recording episode {episode_index} (session {episode_idx + 1}/{cfg.dataset.num_episodes}) ---")

                # Main record loop
                _record_episode(
                    follower, leader, pipeline, dataset, events,
                    cfg.dataset.task, fps, cfg.dataset.episode_time_s,
                )

                if events["rerecord_episode"]:
                    print("Re-recording episode (discarding last take)...")
                    if cfg.dataset.use_tts:
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
                if cfg.dataset.reset_time_s > 0 and episode_idx < cfg.dataset.num_episodes:
                    if cfg.dataset.use_tts:
                        log_say("Reset the environment", True)
                    _reset_phase(follower, leader, pipeline, fps, cfg.dataset.reset_time_s)

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

        print(f"Dataset finalized -> {cfg.dataset.repo_id}")


if __name__ == "__main__":
    main()