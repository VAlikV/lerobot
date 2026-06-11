"""Record UR10 demonstrations — relative absolute target format.

Action / observation schema
===========================
Recorded action (5-D, per frame):
    [x.pos, y.pos, z.pos, yaw.pos, gripper.pos]
    - x/y/z  : env.target_xyz - initial_tcp_xyz  (metres, relative to per-episode home)
    - yaw.pos: env.target_yaw offset from R_home  (radians, already relative)
    - gripper: 1.0 = open, 0.0 = closed

Recorded observation.state (17-D, per frame):
    [joint_pos(6), joint_vel(6), tcp_xyz_rel(3), yaw_offset(1), gripper(1)]
    Same as the HILSERL dataset layout — no swap to the 11-D v2 state.

Why relative instead of absolute or delta
==========================================
- Deltas: ~90 % are near-zero (bang-bang gamepad at 10 Hz). ACT learns mean ≈ 0.
- Absolute (v2): meaningful, but requires recording at 30 Hz with a separate env
  config, and evaluation must know base-frame coordinates.
- Relative: always informative (even "hold still" = "stay at position X"), units
  match the state's tcp_xyz_rel, and the eval offset is a single addition of the
  per-episode initial_tcp_xyz captured at reset — handled by eval_ur10_act_v2_hilserl.py.

Compatible pipeline
===================
    Record  → record_ur10_act_relative.py   (this script)
    Train   → train_ur10_act_v2.py
    Eval    → eval_ur10_act_v2_hilserl.py

To record at 30 Hz with 11-D state instead, see record_ur10_act_v2.py.

Usage
=====
    python act_train/record_ur10_act_relative.py
"""

from __future__ import annotations

import json
import logging
import time

import draccus
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.robots import ur10 as _ur10_register  # noqa: F401
from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from _ur10_reset import auto_reset_to_home

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- user-tunable ---------------------------------------------------------------
CONFIG_PATH = "src/lerobot/rl/ur10_env_3cams_yaw_hilserl.json"
REPO_ID = "local/pcb_act_3cams_yaw_relative_2finetune"
TASK_DESCRIPTION = "pcb_act_3cams_yaw_relative_2finetune"
NUM_EPISODES = 20
EPISODE_TIME_S = 20      # truncates episode if operator doesn't end it early
RESET_TIME_S = 5         # motion-only budget for auto_reset_to_home (env.reset handles the grip window)
RESET_SPEED_MPS = 0.1    # auto-reset linear velocity, m/s
FPS = 10                 # must equal cfg.env.fps
DEVICE = "cpu"

USE_TTS = True
USE_RERUN = True
RERUN_EVERY_N_STEPS = 1
# -------------------------------------------------------------------------------

# Action dimension names — must match what eval_ur10_act_v2_hilserl.py / make_robot_action expect.
ACTION_NAMES: list[str] = ["x.pos", "y.pos", "z.pos", "yaw.pos", "gripper.pos"]


def _build_features(transition_obs: dict) -> dict:
    """Build the LeRobotDataset feature schema.

    Action is fixed 5-D relative target. State and image shapes are read from
    the live transition so they auto-size to whatever env_processor produces
    (17-D state, 128×128 images after crop/resize).
    """
    features: dict[str, dict] = {
        ACTION: {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }
    for key, val in transition_obs.items():
        if "image" in key and isinstance(val, torch.Tensor):
            features[key] = {
                "dtype": "video",
                "shape": tuple(val.squeeze(0).shape),
                "names": ["channels", "height", "width"],
            }
        elif key == OBS_STATE and isinstance(val, torch.Tensor):
            features[key] = {
                "dtype": "float32",
                "shape": tuple(val.squeeze(0).shape),
                "names": None,
            }
    return features


def _relative_target_action(env, *, use_yaw: bool, use_gripper: bool) -> torch.Tensor:
    """Return the env's current target pose RELATIVE to the per-episode home.

    env.target_xyz is the absolute base-frame target the robot is currently
    chasing.  Subtracting env.robot._initial_tcp_xyz (captured at reset) gives
    a per-episode-relative displacement — the same convention the
    hilserl_to_act.py converter produces from HILSERL replay buffers.

    This signal is always meaningful:
    - When the operator commands motion: target drifts away from home → non-zero.
    - When the operator holds still:     target stays put → a position, not zero.
    """
    target_xyz = env.target_xyz
    if target_xyz is None:
        target_xyz = np.array(env.robot.get_current_tcp()[:3], dtype=np.float32)

    initial_xyz = env.robot._initial_tcp_xyz[:3]  # set by env.reset() / capture_baselines()
    target_xyz_rel = np.asarray(target_xyz[:3], dtype=np.float32) - initial_xyz.astype(np.float32)

    target_yaw = float(env.target_yaw) if (use_yaw and env.target_yaw is not None) else 0.0
    gripper_state = float(env.robot.gripper.is_open)

    return torch.tensor(
        [target_xyz_rel[0], target_xyz_rel[1], target_xyz_rel[2], target_yaw, gripper_state],
        dtype=torch.float32,
    )


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    assert cfg.env.fps == FPS, (
        f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})"
    )

    dt = 1.0 / FPS

    # Pre-initialise so the finally block is always safe to reference these,
    # even if make_robot_env or env.reset() raise before assignment.
    env = None
    teleop_device = None
    dataset = None
    episode_idx = 0

    # The try block starts HERE — before make_robot_env — so that any exception
    # (including Ctrl+C during the 15-second grip window inside env.reset()) is
    # guaranteed to reach `finally: env.close()` and avoid leaving RTDE
    # registers locked on the robot.
    try:
        env, teleop_device = make_robot_env(cfg.env)
        env_processor, action_processor = make_processors(env, teleop_device, cfg.env, DEVICE)

        use_gripper = (
            cfg.env.processor.gripper.use_gripper
            if cfg.env.processor.gripper is not None
            else True
        )
        ik_cfg = cfg.env.processor.inverse_kinematics
        use_yaw = bool(getattr(ik_cfg, "use_yaw", False)) if ik_cfg else False

        action_dim_delta = 3 + int(use_yaw) + int(use_gripper)
        neutral_action = torch.zeros(action_dim_delta, dtype=torch.float32)
        if use_gripper:
            neutral_action[-1] = 1.0  # STAY

        obs, info = env.reset()
        env_processor.reset()
        action_processor.reset()
        transition = env_processor(create_transition(observation=obs, info=info))

        features = _build_features(transition[TransitionKey.OBSERVATION])
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=FPS,
            features=features,
            robot_type="ur10",
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
        )
        logger.info(
            "Recording %d episodes (≤%.0f s each) at %d Hz into %s",
            NUM_EPISODES, EPISODE_TIME_S, FPS, REPO_ID,
        )
        logger.info("Action schema (relative): %s", ACTION_NAMES)
        logger.info("State: 17-D HILSERL layout [joint_pos(6), joint_vel(6), tcp_xyz_rel(3), yaw(1), grip(1)]")

        if USE_RERUN:
            init_rerun(session_name=f"ur10_record_relative_{TASK_DESCRIPTION}")

        if USE_TTS:
            log_say(f"Recording episode 1 of {NUM_EPISODES}")

        max_episode_steps = int(EPISODE_TIME_S * FPS)
        episode_step = 0
        global_step = 0
        episode_start = time.perf_counter()

        while episode_idx < NUM_EPISODES:
            step_start = time.perf_counter()

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            terminated = bool(transition.get(TransitionKey.DONE, False))
            truncated = bool(transition.get(TransitionKey.TRUNCATED, False))
            if not terminated and not truncated and episode_step + 1 >= max_episode_steps:
                truncated = True

            # Record the relative absolute target — the key departure from v1/v2.
            action_to_record = _relative_target_action(
                env, use_yaw=use_yaw, use_gripper=use_gripper
            )

            # Pull observation tensors (drop batch dim added by env_processor).
            frame: dict = {ACTION: action_to_record, "task": TASK_DESCRIPTION}
            for k, v in transition[TransitionKey.OBSERVATION].items():
                if isinstance(v, torch.Tensor) and k in features:
                    frame[k] = v.squeeze(0).detach().cpu()
            dataset.add_frame(frame)
            episode_step += 1
            global_step += 1

            if USE_RERUN and (global_step % RERUN_EVERY_N_STEPS == 0):
                rr_obs = {
                    k: v.numpy() for k, v in frame.items()
                    if k not in (ACTION, "task") and isinstance(v, torch.Tensor)
                }
                rr_action = {ACTION: action_to_record.numpy()}
                log_rerun_data(observation=rr_obs, action=rr_action, compress_images=False)

            if terminated or truncated:
                ep_time = time.perf_counter() - episode_start
                rerecord = transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False)
                success = transition[TransitionKey.INFO].get(TeleopEvents.SUCCESS, False)

                if rerecord:
                    logger.info("Re-recording episode %d (%.1fs)", episode_idx + 1, ep_time)
                    dataset.clear_episode_buffer()
                    if USE_TTS:
                        log_say(f"Re-recording episode {episode_idx + 1}")
                else:
                    logger.info(
                        "Episode %d %s after %d steps (%.1fs)",
                        episode_idx + 1,
                        "SUCCESS" if success else "DONE",
                        episode_step,
                        ep_time,
                    )
                    dataset.save_episode()
                    episode_idx += 1

                if episode_idx >= NUM_EPISODES:
                    break

                # Step 1: gracefully stop servoL and move back to nominal home
                # without triggering servoStop (the well-known wedge site).
                auto_reset_to_home(env, dt, RESET_TIME_S, RESET_SPEED_MPS, FPS)

                # Step 2: full env.reset() — applies position randomisation, opens
                # the grip window (reset_time_s from the JSON config, default 15 s)
                # so the operator can re-grip the PCB, then calls capture_baselines()
                # to anchor the next episode's relative actions to the new home pose.
                obs, info = env.reset()
                env_processor.reset()
                action_processor.reset()
                transition = env_processor(create_transition(observation=obs, info=info))
                episode_step = 0
                episode_start = time.perf_counter()
                if USE_TTS:
                    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            precise_sleep(max(dt - (time.perf_counter() - step_start), 0.0))

    except KeyboardInterrupt:
        logger.info("Recording stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("Recording failed")
    finally:
        if dataset is not None:
            try:
                dataset.finalize()
                logger.info("Dataset finalized → %s", REPO_ID)
            except Exception:
                logger.exception("dataset.finalize failed")
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
        logger.info("Recorded %d episodes total", episode_idx)


if __name__ == "__main__":
    main()
