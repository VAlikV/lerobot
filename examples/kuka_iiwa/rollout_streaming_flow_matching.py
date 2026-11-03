"""Execute SFP on KUKA with one integrated action per control tick.

This script connects hardware only from main(). For offline replay use --dry-run;
that path loads recorded observations and never constructs a robot or camera.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.streaming_flow_matching.modeling_streaming_flow_matching import (
    StreamingFlowMatchingPolicy,
)
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_to_transition, transition_to_robot_action
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import KukaJointBoundsAndSafety
from lerobot.utils.robot_utils import precise_sleep

MODEL_PATH = "outputs/kuka_streaming_flow_matching/checkpoints/last/pretrained_model_ema"
DATASET_ID = "local/kuka_sfp"
CONFIG_PATH = Path(__file__).parent / "configs/record_config.json"


def make_safety_pipeline(joint_offset_deg: float = 2.0) -> RobotProcessorPipeline:
    return RobotProcessorPipeline(
        steps=[KukaJointBoundsAndSafety(joint_offset_deg=joint_offset_deg)],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )


def predict_robot_action(policy, preprocessor, postprocessor, frame, features, safety):
    """Shared hardware/replay path: processors -> streaming tick -> physical action."""
    action = postprocessor(policy.select_action(preprocessor(frame)))
    if not torch.isfinite(action).all():
        raise ValueError("SFP produced a non-finite robot action")
    command = make_robot_action(action, features)
    return safety(command)


def validate_kuka_contract(policy, metadata) -> None:
    cfg = policy.config
    if cfg.state_feature_names != metadata.features["observation.state"]["names"]:
        raise ValueError("Dataset state feature order differs from checkpoint")
    if cfg.action_feature_names != metadata.features["action"]["names"]:
        raise ValueError("Dataset action feature order differs from checkpoint")
    expected = [f"joint_{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
    if cfg.action_feature_names != expected:
        raise ValueError("This rollout requires seven absolute joints in degrees and gripper.pos")
    if list(cfg.image_features) != list(metadata.camera_keys):
        raise ValueError("Dataset camera order differs from checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=None, help="1 = fresh feedback every tick")
    parser.add_argument("--integration-steps-per-action", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Replay dataset; do not connect hardware")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    metadata = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    policy = StreamingFlowMatchingPolicy.from_pretrained(
        args.model, cli_overrides=[f"--device={args.device}"]
    )
    policy.config.device = args.device
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
    if args.integration_steps_per_action is not None:
        policy.config.integration_steps_per_action = args.integration_steps_per_action
    policy.config.__post_init__()
    validate_kuka_contract(policy, metadata)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    safety = make_safety_pipeline()
    if args.dry_run:
        dataset = LeRobotDataset(args.dataset, root=args.dataset_root)
        last_episode = None
        for i in range(min(len(dataset), args.max_steps)):
            sample = dataset[i]
            episode = int(sample["episode_index"])
            if episode != last_episode:
                policy.reset()
                pre.reset()
                post.reset()
                last_episode = episode
            frame = {key: sample[key].unsqueeze(0) for key in policy.config.input_features}
            command = predict_robot_action(policy, pre, post, frame, metadata.features, safety)
            if i % 30 == 0:
                print(f"frame={i} command={command}")
        return

    # Importing these modules does not connect hardware. Construction/connect happen
    # only on the explicitly selected hardware path, after all model checks pass.
    import draccus
    from record_config import RecordingConfig

    from lerobot.robots.kuka_iiwa import KukaIiwa

    with args.config.open() as f:
        recording = draccus.decode(RecordingConfig, json.load(f))
    if recording.robot.use_task_space or not recording.robot.use_direct_joint_control:
        raise ValueError("SFP KUKA rollout requires direct absolute joint control")
    if recording.fps != metadata.fps:
        raise ValueError("Recording/control FPS must match the training dataset")
    if set(recording.robot.cameras) != {
        key.removeprefix("observation.images.") for key in policy.config.image_features
    }:
        raise ValueError("Robot cameras differ from checkpoint")
    safety = make_safety_pipeline(recording.pipeline.joint_offset_deg)
    follower = KukaIiwa(recording.robot)
    if list(follower.action_features) != policy.config.action_feature_names:
        raise ValueError("Robot action order differs from checkpoint")
    if [
        k for k, value in follower.observation_features.items() if value is float
    ] != policy.config.state_feature_names:
        raise ValueError("Robot state order differs from checkpoint")
    period = 1 / metadata.fps
    try:
        follower.connect()
        for episode in range(args.episodes):
            # Each episode starts at the robot's current measured pose. No automatic homing.
            policy.reset()
            pre.reset()
            post.reset()
            safety.reset()
            for _ in range(args.max_steps):
                start = time.perf_counter()
                frame = build_inference_frame(
                    observation=follower.get_observation(), ds_features=metadata.features, device=args.device
                )
                command = predict_robot_action(policy, pre, post, frame, metadata.features, safety)
                follower.send_action(command)
                elapsed = time.perf_counter() - start
                if elapsed > period:
                    logging.warning("SFP control tick %.1f ms exceeds %.1f ms", elapsed * 1000, period * 1000)
                precise_sleep(max(0, period - elapsed))
            print(f"Episode {episode + 1} finished")
    except KeyboardInterrupt:
        print("Stopping SFP rollout")
    finally:
        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":
    main()
