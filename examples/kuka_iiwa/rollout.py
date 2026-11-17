import argparse
import json
import time

import draccus
import torch
from collections import deque

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.processor import (
    RobotAction,
    RobotProcessorPipeline,
    RelativeActionsProcessorStep,
    AbsoluteActionsProcessorStep
)
from lerobot.processor.converters import (
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.kuka_iiwa import KukaIiwa
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import KukaJointBoundsAndSafety
from lerobot.common.control_utils import follower_smooth_move_to
from lerobot.utils.constants import OBS_STR
from lerobot.utils.robot_utils import precise_sleep

from record_config import RecordingConfig

def load_config(path: str) -> RecordingConfig:
    with open(path, "r") as f:
        raw_cfg = json.load(f)
    return draccus.decode(RecordingConfig, raw_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ACT policy on KUKA.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration.")
    parser.add_argument("--dataset-id", required=True, help="Dataset repo ID for policy features and statistics.")
    parser.add_argument("--model-id", required=True, help="Model repo ID or local checkpoint path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.rollout.device if torch.cuda.is_available() or cfg.rollout.device == "cpu" else "cpu")
    fps = cfg.fps

    reset_pose = cfg.dataset.reset_pose

    dataset_metadata = LeRobotDatasetMetadata(args.dataset_id)

    policy_config = ACTConfig.from_pretrained(args.model_id)
    policy_config.temporal_ensemble_coeff = cfg.rollout.temporal_ensemble_coeff
    policy_config.n_action_steps = 1 if cfg.rollout.temporal_ensemble_coeff is not None else cfg.rollout.n_action_steps
    policy = ACTPolicy.from_pretrained(args.model_id, config=policy_config)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model_id,
        dataset_stats=dataset_metadata.stats,
    )

    use_relative_actions = policy.config.use_relative_actions

    safety_pipeline = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            KukaJointBoundsAndSafety(joint_offset_deg=cfg.pipeline.joint_offset_deg),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    follower = KukaIiwa(cfg.robot)
    follower.connect()

    action_queue = deque()

    dt_s = 1.0 / float(fps)
    print(f"Rolling out {args.model_id} on {args.dataset_id} action space, relative_actions={policy.config.use_relative_actions} fps={fps}")

    try:
        for episode_idx in range(cfg.rollout.max_episodes):
            if cfg.rollout.reset_to_pose:
                current = follower.get_observation()
                target = dict(zip(follower.action_features.keys(), reset_pose))
                follower_smooth_move_to(follower, current, target, duration_s=3.0)

            policy.reset()
            action_queue.clear()
            print(f"\n--- Episode {episode_idx + 1}/{cfg.rollout.max_episodes} ---")

            for step_idx in range(cfg.rollout.max_steps_per_episode):
                start_t = time.perf_counter()

                if not use_relative_actions:

                    obs = follower.get_observation()
                    observation_frame = build_inference_frame(
                        observation=obs, ds_features=dataset_metadata.features, device=device
                    )
                    observation = preprocess(observation_frame)

                    with torch.no_grad():
                        action = policy.select_action(observation)
                    action = postprocess(action)

                elif policy.config.temporal_ensemble_coeff is not None:
                    obs = follower.get_observation()
                    observation_frame = build_inference_frame(
                        observation=obs, ds_features=dataset_metadata.features, device=device
                    )
                    observation = preprocess(observation_frame)
                    with torch.no_grad():
                        chunk = policy.predict_action_chunk(observation)
                    # Relative chunks must share absolute coordinates before averaging.
                    chunk = postprocess(chunk)
                    action = policy.temporal_ensembler.update(chunk)

                else:

                    if len(action_queue) == 0:
                        obs = follower.get_observation()
                        observation_frame = build_inference_frame(
                            observation=obs, ds_features=dataset_metadata.features, device=device
                            )
                        observation = preprocess(observation_frame)

                        with torch.no_grad():
                            action = policy.predict_action_chunk(observation)
                        chunk = action[:, :cfg.rollout.n_action_steps]
                        chunk = postprocess(chunk)
                        action_queue.extend(chunk.squeeze(0))
                    
                    action = action_queue.popleft().unsqueeze(0)
                
                robot_action = make_robot_action(action, dataset_metadata.features)
                robot_action = safety_pipeline(robot_action)
                follower.send_action(robot_action)

                precise_sleep(max(dt_s - (time.perf_counter() - start_t), 0.0))

            print(f"Episode {episode_idx + 1} finished.")

    except KeyboardInterrupt:
        print("\nStopping rollout...")

    finally:
        if follower.is_connected:
            follower.disconnect()


if __name__ == "__main__":
    main()
