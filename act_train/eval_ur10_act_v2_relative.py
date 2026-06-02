"""Closed-loop inference of an ACT policy trained on the HILSERL-converted dataset.

How this differs from eval_ur10_act_v2.py
==========================================
The policy here was trained on the HILSERL dataset converted by
``hilserl_to_act.py``.  During conversion, delta actions are turned into
*relative absolute targets*:

    action = [tcp_x_rel + delta_x,  tcp_y_rel + delta_y,  tcp_z_rel + delta_z,
              yaw_offset + delta_yaw,  gripper_binary]

where ``tcp_xyz_rel`` and ``yaw_offset`` are already in the 17-D state vector
(relative to the per-episode initial pose captured by the env at reset).

At inference we therefore:
  1. Use the *same* 17-D HILSERL state from env_processor (no swap to 11-D).
  2. Add ``env.robot._initial_tcp_xyz`` to the policy's xyz output to convert
     the relative target back to absolute base-frame before calling
     ``set_act_target()`` (which expects absolute xyz).

Usage
=====
    python act_train/eval_ur10_act_v2_hilserl.py
"""

from __future__ import annotations

import json
import logging
import time

import draccus
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.robots import ur10 as _ur10_register  # noqa: F401
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_processors, make_robot_env
from lerobot.utils.constants import OBS_STATE
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

from _ur10_reset import auto_reset_to_home  # sibling module in act_train/

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- user-tunable ---------------------------------------------------------------
MODEL_DIR = "outputs/act/ur10/pcb_act_3cams_yaw_relative_finetune/step_10000"
DATASET_REPO_ID = "local/pcb_act_3cams_yaw_relative_finetune"   # for dataset stats (normalization)
CONFIG_PATH = "src/lerobot/rl/ur10_env_3cams_yaw_hilserl.json"
NUM_EPISODES = 30
EPISODE_TIME_S = 20      # safety upper bound; user ends earlier via gamepad
RESET_TIME_S = 5         # motion-only budget for auto_reset_to_home (env.reset handles the grip window)
RESET_SPEED_MPS = 0.1    # auto-reset linear velocity, m/s (matches env.reset's moveL)
FPS = 10
# -------------------------------------------------------------------------------


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    assert cfg.env.fps == FPS, (
        f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})"
    )

    dt = 1.0 / FPS

    # ---- policy + processors ------------------------------------------------
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    policy: ACTPolicy = ACTPolicy.from_pretrained(MODEL_DIR)
    policy.eval()
    device = torch.device(policy.config.device)
    policy.to(device)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=MODEL_DIR, dataset_stats=metadata.stats
    )
    logger.info("Policy loaded from %s on %s", MODEL_DIR, device)
    logger.info(
        "ACT config: chunk_size=%d, n_action_steps=%d, temporal_ensemble_coeff=%s",
        policy.config.chunk_size,
        policy.config.n_action_steps,
        policy.config.temporal_ensemble_coeff,
    )

    # Pre-initialise so the finally block can always reference them safely,
    # even if make_robot_env or env.reset() raise before they are assigned.
    env = None
    teleop_device = None
    episode_idx = 0

    # ---- env + processors --------------------------------------------------
    # The try block starts HERE — before make_robot_env — so that any exception
    # (including KeyboardInterrupt during the 15-second grip window inside
    # env.reset()) is guaranteed to reach `finally: env.close()`.  Without this,
    # a Ctrl+C during the initial reset leaves the RTDE TCP connections open on
    # the robot and the next run fails with "RTDE input registers already in use".
    try:
        env, teleop_device = make_robot_env(cfg.env)
        env_processor, _action_processor = make_processors(env, teleop_device, cfg.env, str(device))
        obs, _info = env.reset()
        env_processor.reset()

        initial_tcp_xyz = torch.tensor(
            env.robot._initial_tcp_xyz[:3], dtype=torch.float32, device=device
        )

        def _build_obs_for_policy() -> dict:
            """Pull a single observation through env_processor (cropped images +
            17-D HILSERL state). Returns the obs dict ready for policy.select_action."""
            raw = env._augment_observation(env.robot.get_observation())
            tr = env_processor(create_transition(
                observation=raw, info={TeleopEvents.IS_INTERVENTION: False},
            ))
            return {
                k: v for k, v in tr[TransitionKey.OBSERVATION].items()
                if k in policy.config.input_features
            }

        episode_step = 0
        episode_start = time.perf_counter()
        logger.info(
            "Inference at %d Hz for %d episodes. Triangle/Cross = success/fail; Ctrl+C to exit.",
            FPS, NUM_EPISODES,
        )
        logger.info("--- Episode %d ---", episode_idx + 1)

        while episode_idx < NUM_EPISODES:
            t0 = time.perf_counter()

            # 1. Build observation (17-D HILSERL state + cropped images).
            obs_batch = _build_obs_for_policy()

            # 2. Normalize, predict, unnormalize.
            obs_batch = preprocess(obs_batch)
            with torch.no_grad():
                action_tensor = policy.select_action(obs_batch)
            action_tensor = postprocess(action_tensor)

            # 3. The policy outputs xyz RELATIVE to the per-episode initial TCP
            #    pose (matching the dataset's action convention).  Convert to
            #    absolute base-frame before handing to set_act_target().
            action_tensor = action_tensor.clone()
            action_tensor[0, :3] += initial_tcp_xyz.to(action_tensor.device)

            # 4. Build named dict matching action schema (x.pos, y.pos, …).
            action_dict = make_robot_action(action_tensor, metadata.features)

            # 5. Drive the robot. Bounds / rotation / gripper logic all live
            #    inside set_act_target, so nothing extra needed here.
            env.set_act_target(action_dict)

            episode_step += 1
            if episode_step % 10 == 0:
                logger.info(
                    "  step %d  target=[x=%+.4f y=%+.4f z=%+.4f yaw=%+.4f g=%.2f]",
                    episode_step,
                    float(action_dict.get("x.pos", 0.0)),
                    float(action_dict.get("y.pos", 0.0)),
                    float(action_dict.get("z.pos", 0.0)),
                    float(action_dict.get("yaw.pos", 0.0)),
                    float(action_dict.get("gripper.pos", 0.0)),
                )

            # Episode termination: client-side time limit (the env's TimeLimit
            # processor isn't in this loop because we're not going through
            # env_processor at all in the v2 eval). Gamepad-driven success/fail
            # via teleop events still works because make_robot_env connects the
            # gamepad — we read its events here.
            truncated = episode_step >= int(EPISODE_TIME_S * FPS)
            success = False
            terminate = False
            if teleop_device is not None:
                events = teleop_device.get_teleop_events()
                success = bool(events.get(TeleopEvents.SUCCESS, False))
                terminate = bool(events.get(TeleopEvents.TERMINATE_EPISODE, False))
            done = success or terminate

            if done or truncated:
                ep_time = time.perf_counter() - episode_start
                status = "SUCCESS" if success else ("TERMINATED" if terminate else "TIMEOUT")
                logger.info(
                    "Episode %d %s after %d steps (%.1fs)",
                    episode_idx + 1, status, episode_step, ep_time,
                )
                episode_idx += 1
                if episode_idx >= NUM_EPISODES:
                    break

                # Step 1: gracefully stop servoL and return to nominal home
                # without triggering the servoStop freeze.
                auto_reset_to_home(env, dt, RESET_TIME_S, RESET_SPEED_MPS, FPS)

                # Step 2: full env.reset() — randomises position, opens grip
                # window (reset_time_s from config) so operator can re-grip,
                # and calls capture_baselines() for the new episode's home pose.
                env.reset()

                # Re-read initial_tcp_xyz: home is randomised so each episode
                # has a different baseline for the relative→absolute conversion.
                initial_tcp_xyz = torch.tensor(
                    env.robot._initial_tcp_xyz[:3], dtype=torch.float32, device=device
                )

                env_processor.reset()
                policy.reset()
                episode_step = 0
                episode_start = time.perf_counter()
                logger.info("--- Episode %d ---", episode_idx + 1)

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        logger.info("Inference stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("Inference failed")
    finally:
        logger.info("Completed %d episodes", episode_idx)
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logger.exception("teleop disconnect failed")


if __name__ == "__main__":
    main()
