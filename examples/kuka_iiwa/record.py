"""
KUKA Leader -> KUKA IIWA Follower Teleoperation + Dataset Recording

Keyboard Controls During Recording:
--> / n - Prematurely terminate the current episode / reset; proceed to the next. 
<-- / r - Cancel the current episode; re-record.
esc / q - Immediately stop the session.
      s - Scale robot actions
"""

import json
import time
import draccus

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility, follower_smooth_move_to
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

from lerobot.robots.kuka_iiwa import KukaIiwa
from lerobot.teleoperators.kuka_leader import KukaLeader
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import (
    GripperPositionToDiscrete,
    KukaJointBoundsAndSafety,
    KukaJointDeltaScaling,
)
from record_config import (
    RecordingConfig
)
from keyboard_controls import init_kuka_keyboard_listener
from record_utils import _finish_episode_buffer, _assert_pending_episode_indices

CONFIG_PATH = "examples/kuka_iiwa/configs/record_config.json"

# Main record episode loop
# Optionally can be switched to record_loop from lerobot.scripts.lerobot_record
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
    joint_scaler
) -> None:
    episode_start = time.perf_counter()
    while time.perf_counter() - episode_start < duration_s:
        t0 = time.perf_counter()

        obs = follower.get_observation()

        joint_scaler.set_enabled(events["apply_scale"])

        leader_action = leader.get_action()
        leader_action = pipeline((leader_action, obs))

        sent_action = follower.send_action(leader_action)

        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
        dataset.add_frame({**observation_frame, **action_frame, "task": task})

        if events["exit_early"]:
            events["exit_early"] = False
            break

        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


# Reset with optional linear interpolation to starting pose and teleoperation without recording
def _reset_phase(
    follower: KukaIiwa,
    leader: KukaLeader,
    pipeline: RobotProcessorPipeline,
    events: dict,
    fps: int,
    reset_to_pose: bool,
    reset_pose: list[float],
    duration_s: float,
    joint_scaler
) -> None:

    if reset_to_pose:
        current = follower.get_observation()
        target = dict(zip(follower.action_features.keys(), reset_pose))
        follower_smooth_move_to(robot=follower, current=current, target=target, duration_s=3.0)

        input("\nPress Enter to sync leader to follower home position...")
        
        follower_obs = follower.get_observation()
        follower_obs["gripper.pos"] = 45.0

        leader.send_goal_position(
            follower_obs,
            hold_s=3.0,
            tolerance_counts=50,
        )

        # Stop applying scale
        joint_scaler.reset()
        events["apply_scale"] = False
        leader.reset_filter()
    
    reset_start = time.perf_counter()
    while time.perf_counter() - reset_start < duration_s:
        t0 = time.perf_counter()

        obs = follower.get_observation()
        joint_scaler.set_enabled(events["apply_scale"])
        leader_action = leader.get_action()
        leader_action = pipeline((leader_action, obs))
        follower.send_action(leader_action)

        if events["exit_early"]:
            events["exit_early"] = False
            break

        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def load_config(path: str) -> RecordingConfig:
    with open(path, "r") as f:
        raw_cfg = json.load(f)

    return draccus.decode(
        RecordingConfig,
        raw_cfg,
    )


def _log_say_console(text: str, play_sound: bool) -> None:
    """Print TTS text explicitly, then pass it to the existing voice helper."""
    print(f"[TTS] {text}", flush=True)
    log_say(text, play_sound)


def main():

    cfg = load_config(CONFIG_PATH)

    fps = cfg.fps

    follower = KukaIiwa(cfg.robot)
    leader = KukaLeader(cfg.leader)

    joint_scaler = KukaJointDeltaScaling(scale_factor=cfg.pipeline.scale_factor,)

    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            joint_scaler,
            KukaJointBoundsAndSafety(
                joint_offset_deg=cfg.pipeline.joint_offset_deg,
            ),
            GripperPositionToDiscrete(
                threshold=cfg.pipeline.gripper_threshold,
                reverse=cfg.pipeline.gripper_reverse,
            )
        ],
        to_transition=robot_action_observation_to_transition,
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
            streaming_encoding=False,
        )

    follower.connect()
    leader.connect()

    listener, events = init_kuka_keyboard_listener()

    pending_episode_buffers: list[dict] = []

    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        # Sync
        print("Reading follower home position...")
        
        _log_say_console(f"Press Enter to sync", cfg.dataset.use_tts)
        input("\nPress Enter to sync leader to follower home position...")

        follower_obs = follower.get_observation()
        follower_obs["gripper.pos"] = 45.0

        leader.send_goal_position(
            follower_obs,
            hold_s=3.0,
            tolerance_counts=50,
        )

        print("Starting record loop...")


        episode_idx = 0
        while episode_idx < cfg.dataset.num_episodes and not events["stop_recording"]:

            _log_say_console(f"Recording episode {episode_idx}", cfg.dataset.use_tts)

            # Main record loop
            _record_episode(
                follower, leader, pipeline, dataset, events,
                cfg.dataset.task, fps, cfg.dataset.episode_time_s,
                joint_scaler
            )

            # Reset if not stopped or last episode
            if not events["stop_recording"] and (
                (episode_idx < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                _log_say_console("Reset the environment", cfg.dataset.use_tts)
                _reset_phase(
                    follower, leader, pipeline, events, fps, 
                    cfg.dataset.reset_to_pose, cfg.dataset.reset_pose, cfg.dataset.reset_time_s,
                    joint_scaler
                )

            if events["rerecord_episode"]:
                print("Re-recording episode (discarding last take)...")
                _log_say_console("Re-record episode", cfg.dataset.use_tts)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                _finish_episode_buffer(dataset, rerecord=True)
                continue

            # Save episode
            pending_episode_buffers.append(_finish_episode_buffer(dataset, rerecord=False))
            episode_idx += 1

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")

    finally:

        if leader.is_connected:
            leader.disconnect()

        if follower.is_connected:
            follower.disconnect()

        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()

        _assert_pending_episode_indices(dataset, pending_episode_buffers)
        for idx, episode_buffer in enumerate(pending_episode_buffers, start=1):
            print(f"Saving episode {idx}/{len(pending_episode_buffers)} after disconnect")
            dataset.save_episode(episode_data=episode_buffer)

        if dataset:
            dataset.finalize()

        if listener is not None:
            listener.stop()

        print(f"Dataset finalized -> {cfg.dataset.repo_id}")


if __name__ == "__main__":
    main()
