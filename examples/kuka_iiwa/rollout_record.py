"""Record ACT rollouts as local LeRobot episodes; never connect hardware on import."""

import argparse
import json
import logging
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import torch

from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.feature_utils import build_dataset_frame, dataset_to_policy_features
from lerobot.utils.robot_utils import precise_sleep


def validate_execution(config):
    """Avoid averaging relative chunks anchored at different robot states."""
    if config.use_relative_actions and config.temporal_ensemble_coeff is not None:
        raise ValueError("Temporal ensembling of relative actions is unsupported; use absolute ACT")


def record_episode(
    robot, policy, pre, post, safety, dataset, features, events, task, fps, duration_s, device, max_steps=None
):
    """Record the observation before each command and the action returned by the driver."""
    queue = deque()
    start = time.perf_counter()
    step = 0
    while not events["stop_recording"] and time.perf_counter() - start < duration_s:
        if events.get("exit_early") or events.get("rerecord_episode"):
            break
        tick = time.perf_counter()
        obs = robot.get_observation()
        frame = build_dataset_frame(dataset.features, obs, prefix="observation")
        # Copy before send_action: camera/driver buffers may be reused.
        frame = {key: value.copy() for key, value in frame.items()}
        with torch.inference_mode():
            if policy.config.use_relative_actions:
                if not queue:
                    observation = pre(
                        build_inference_frame(observation=obs, ds_features=features, device=device)
                    )
                    chunk = policy.predict_action_chunk(observation)[:, : policy.config.n_action_steps]
                    queue.extend(post(chunk).squeeze(0))
                action = queue.popleft().unsqueeze(0)
            else:
                observation = pre(build_inference_frame(observation=obs, ds_features=features, device=device))
                action = post(policy.select_action(observation))
        if not torch.isfinite(action).all():
            raise ValueError("Policy returned non-finite actions")
        command = safety(make_robot_action(action, features))
        sent = robot.send_action(command)
        dataset.add_frame(
            {**frame, **build_dataset_frame(dataset.features, sent, prefix="action"), "task": task}
        )
        step += 1
        elapsed = time.perf_counter() - tick
        if elapsed > 1 / fps and (step == 1 or step % fps == 0):
            logging.warning("Control tick %.3fs exceeds %.3fs", elapsed, 1 / fps)
        if max_steps is not None and step >= max_steps:
            break
        precise_sleep(max(1 / fps - elapsed, 0))


def parse_args():
    """Parse paths separately for training metadata and output recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/record_config.json")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-time-s", type=float, default=30)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--temporal-ensemble-coeff", help="Number or none; omitted keeps checkpoint setting")
    parser.add_argument("--reset-to-pose", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.num_episodes < 1 or args.episode_time_s <= 0:
        parser.error("Episode count and duration must be positive")
    if args.root.exists():
        parser.error("Output root already exists; choose a new directory (resume is not supported)")
    return args


def cleanup(follower, listener, dataset):
    """Release partially connected hardware, then finalize completed episodes."""
    try:
        follower.disconnect()
    finally:
        try:
            if listener is not None:
                listener.stop()
        finally:
            if dataset is not None:
                try:
                    if dataset.has_pending_frames():
                        dataset.clear_episode_buffer()
                finally:
                    dataset.finalize()


def main():
    """Run episodic recording with bounded streaming video queues and explicit starts."""
    args = parse_args()
    # Hardware-specific imports are deferred so offline tests need no KUKA libraries.
    import draccus
    from keyboard_controls import init_kuka_keyboard_listener
    from record_config import RecordingConfig

    from lerobot.common.control_utils import follower_smooth_move_to
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import RobotProcessorPipeline
    from lerobot.processor.converters import robot_action_to_transition, transition_to_robot_action
    from lerobot.robots.kuka_iiwa import KukaIiwa
    from lerobot.robots.kuka_iiwa.robot_kinematic_processor import KukaJointBoundsAndSafety

    cfg = draccus.decode(RecordingConfig, json.loads(args.config.read_text()))
    if cfg.robot.use_task_space or not cfg.robot.use_direct_joint_control:
        raise ValueError("This rollout requires direct joint control in absolute joint coordinates")
    if not (args.source_dataset / "meta/info.json").is_file():
        raise FileNotFoundError("Source dataset must contain local meta/info.json")
    metadata = LeRobotDatasetMetadata("local/rollout_source", root=args.source_dataset)
    if metadata.fps != cfg.fps:
        raise ValueError("Control FPS must match source dataset FPS")
    config = ACTConfig.from_pretrained(args.policy)
    config.device = args.device
    if args.n_action_steps is not None:
        config.n_action_steps = args.n_action_steps
    if args.temporal_ensemble_coeff is not None:
        config.temporal_ensemble_coeff = (
            None if args.temporal_ensemble_coeff.lower() == "none" else float(args.temporal_ensemble_coeff)
        )
    if not 1 <= config.n_action_steps <= config.chunk_size:
        raise ValueError("n_action_steps must be in [1, chunk_size]")
    if config.temporal_ensemble_coeff is not None and config.n_action_steps != 1:
        raise ValueError("Temporal ensembling requires n_action_steps=1")
    validate_execution(config)
    expected = dataset_to_policy_features(metadata.features)
    for key, feature in (config.input_features | config.output_features).items():
        if key not in expected or expected[key] != feature:
            raise ValueError(f"Checkpoint/source dataset feature mismatch: {key}")
    for field, names in [
        ("action_feature_names", metadata.features["action"]["names"]),
        ("state_feature_names", metadata.features["observation.state"]["names"]),
    ]:
        saved_names = getattr(config, field, None)
        if saved_names is not None and saved_names != names:
            raise ValueError(f"Checkpoint/source dataset name order mismatch: {field}")
    policy = ACTPolicy.from_pretrained(args.policy, config=config, strict=True).eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    follower = KukaIiwa(cfg.robot)
    action_names = metadata.features["action"]["names"]
    if list(follower.action_features) != action_names:
        raise ValueError("Robot/source dataset action order mismatch")
    for key in metadata.features:
        if not key.startswith("observation.images."):
            continue
        if key.removeprefix("observation.images.") not in cfg.robot.cameras:
            raise ValueError(f"Missing camera: {key}")
    for key, feature in metadata.features.items():
        if key.startswith("observation.images.") and tuple(feature["shape"][:2]) != tuple(
            reversed(cfg.robot.resolution)
        ):
            raise ValueError(f"Robot image resolution differs from source dataset: {key}")
    if len(cfg.dataset.reset_pose) != len(action_names):
        raise ValueError("reset_pose does not match robot actions")
    safety = RobotProcessorPipeline(
        steps=[KukaJointBoundsAndSafety(joint_offset_deg=cfg.pipeline.joint_offset_deg)],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )
    features = {
        key: deepcopy(value)
        for key, value in metadata.features.items()
        if key == "action" or key.startswith("observation.")
    }
    dataset = None
    listener = None
    try:
        dataset = LeRobotDataset.create(
            args.repo_id,
            cfg.fps,
            root=args.root,
            features=features,
            robot_type=follower.name,
            use_videos=True,
            streaming_encoding=True,
            encoder_threads=1,
            encoder_queue_maxsize=4,
            image_writer_processes=0,
            image_writer_threads=0,
        )
        (args.root / "rollout_config.json").write_text(
            json.dumps(
                {
                    "policy": str(args.policy.resolve()),
                    "source_dataset": str(args.source_dataset.resolve()),
                    "n_action_steps": config.n_action_steps,
                    "temporal_ensemble_coeff": config.temporal_ensemble_coeff,
                    "task": cfg.dataset.task,
                    "fps": cfg.fps,
                },
                indent=2,
            )
        )
        follower.connect()
        events = {"stop_recording": False}
        episode = 0
        while episode < args.num_episodes and not events["stop_recording"]:
            if listener is not None:
                listener.stop()
                listener = None
            input("Prepare workspace. Enter starts reset (if enabled) and rollout; Ctrl+C cancels: ")
            listener, events = init_kuka_keyboard_listener()
            if events["stop_recording"]:
                break
            if args.reset_to_pose:
                follower_smooth_move_to(
                    follower,
                    follower.get_observation(),
                    dict(zip(action_names, cfg.dataset.reset_pose, strict=True)),
                    duration_s=3,
                )
            policy.reset()
            pre.reset()
            post.reset()
            events["exit_early"] = events["rerecord_episode"] = False
            record_episode(
                follower,
                policy,
                pre,
                post,
                safety,
                dataset,
                metadata.features,
                events,
                cfg.dataset.task,
                cfg.fps,
                args.episode_time_s,
                args.device,
            )
            if events["rerecord_episode"]:
                dataset.clear_episode_buffer()
            elif dataset.has_pending_frames():
                dataset.save_episode()
                episode += 1
                print(f"Saved episode {episode}/{args.num_episodes}")
    except KeyboardInterrupt:
        print("Interrupted; current partial episode will be discarded.")
    finally:
        cleanup(follower, listener, dataset)


if __name__ == "__main__":
    main()
