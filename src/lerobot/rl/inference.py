#!/usr/bin/env python

"""
Run a trained HIL-SERL model on the real robot
The gamepad can still be used for episode management: Use tringle to restart the episode

Usage:
    python -m lerobot.rl.inference --config_path outputs/train/.../checkpoints/last/pretrained_model/train_config.json
"""

import logging
import sys
import time
from pathlib import Path

import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
    )

logging.basicConfig(level=logging.INFO)


def _resolve_pretrained_dir() -> str:
    """Extract the pretrained_model directory from --config_path CLI arg."""
    for i, arg in enumerate(sys.argv):
        if "config_path" in arg:
            if "=" in arg:
                config_path = Path(arg.split("=", 1)[1])
            elif i + 1 < len(sys.argv):
                config_path = Path(sys.argv[i + 1])
            else:
                break
            # train_config.json sits inside pretrained_model/
            pretrained_dir = config_path.parent
            if pretrained_dir.exists() and (pretrained_dir / "model.safetensors").exists():
                return str(pretrained_dir)
    raise ValueError(
        "Could not determine pretrained model directory. "
        "Pass --config_path pointing to .../pretrained_model/train_config.json"
        )


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    pretrained_dir = _resolve_pretrained_dir()
    logging.info(f"Loading pretrained model from: {pretrained_dir}")

    # -- Set pretrained_path so make_policy loads weights -------------------
    cfg.policy.pretrained_path = pretrained_dir

    # -- Create environment and processors ----------------------------------
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(
        env, teleop_device, cfg.env, cfg.policy.device
    )

    # -- Create and load policy ---------------------------------------------
    policy: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()

    device = cfg.policy.device
    logging.info(f"Policy loaded on {device}")
    logging.info(f"Policy type: {type(policy).__name__}")

    # -- Inference loop -----------------------------------------------------
    fps = cfg.env.fps or 10
    max_episodes = 100
    episode_count = 0

    logging.info(f"Starting inference at {fps} FPS...")
    logging.info("Gamepad controls:")
    logging.info("  Triangle  = mark SUCCESS")
    logging.info("  Cross     = mark FAILURE")
    logging.info("  Ctrl+C      = exit")

    try:
        for episode in range(max_episodes):
            obs, info = env.reset()
            env_processor.reset()
            action_processor.reset()

            # Process initial observation through env pipeline
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

            episode_reward = 0.0
            step_count = 0

            logging.info(f"--- Episode {episode + 1} ---")

            while True:
                t0 = time.perf_counter()

                # Extract observation for policy (only input features)
                observation = {
                    k: v
                    for k, v in transition[TransitionKey.OBSERVATION].items()
                    if k in cfg.policy.input_features
                    }

                # Policy inference
                with torch.no_grad():
                    action = policy.select_action(batch=observation)

                # Step environment through processor pipeline
                new_transition = step_env_and_process_transition(
                    env=env,
                    transition=transition,
                    action=action,
                    env_processor=env_processor,
                    action_processor=action_processor,
                    )

                reward = new_transition[TransitionKey.REWARD]
                done = new_transition.get(TransitionKey.DONE, False)
                truncated = new_transition.get(TransitionKey.TRUNCATED, False)

                episode_reward += float(reward)
                step_count += 1
                transition = new_transition

                if step_count %10 == 0:
                    logging.info(f" step {step_count}, reward={episode_reward}:.2f")

                # Maintain FPS
                elapsed = time.perf_counter() - t0
                precise_sleep(max(1.0 / fps - elapsed, 0.0))

                if done or truncated:
                    logging.info(f"Episode ended: done={done}, truncated={truncated}, step={step_count}")
                    break

            status = "SUCCESS" if episode_reward > 0 else "DONE"
            logging.info(
                f"Episode {episode + 1} finished: {status}, "
                f"reward={episode_reward:.2f}, steps={step_count}"
            )
            episode_count += 1

    except KeyboardInterrupt:
        logging.info("\nInference stopped...")
    except Exception as e:
        logging.exception(f"Inference failed with error: {e}")
    finally:
        logging.info(f"Completed {episode_count} episodes....")
        env.close()
        if teleop_device is not None:
            teleop_device.disconnect()


if __name__ == "__main__":
    main()
