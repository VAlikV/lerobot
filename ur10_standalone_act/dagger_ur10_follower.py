"""HG-DAgger correction collection for the standalone UR10 follower.

This is the standalone counterpart of `act_train/eval_ur10_act.py`'s HG-DAgger
mode (RECORD_DAGGER_DATASET), rebuilt for the direct-robot (no gym_manipulator)
pipeline. It runs a trained ACT policy closed-loop AND lets the operator take
over at any instant with the gamepad to correct the policy's failure modes; the
post-intervention trajectory is recorded into a NEW dataset you then fine-tune on.

How it works each step
======================
  1. The policy proposes a relative-to-home action (same as eval_ur10_follower).
  2. If the operator HOLDS R1 (deadman), the GAMEPAD action overrides the policy
     for that step (the human "corrects"); otherwise the policy drives.
  3. Whichever action actually drove the robot is what gets RECORDED — this is the
     HG-DAgger signal: on-policy states paired with the human's corrective action.
  4. While the policy drives, the teleop shadows it (`teleop.sync_to`) so grabbing
     R1 is a bumpless handoff (no jump from a stale relative target).

The recorded dataset uses the SAME feature schema as record_ur10_follower.py
(11-D state + 5-D relative action + the 3 cropped cameras), built straight from
the robot's features — so a model fine-tuned on it sees an identical action /
observation layout. Frames are built with lerobot's own `build_dataset_frame`,
exactly as `record_loop` does, so the schema can never drift from a recorded set.

Episode buttons (same async listener as record/eval):
    TRIANGLE = end episode (keep), SQUARE = redo (discard), CROSS = stop session.

Fine-tuning loop (HG-DAgger iteration)
======================================
  1. Train a base policy:        train_ur10_follower.py (PRETRAINED_PATH=None)
  2. Collect corrections here:    dagger_ur10_follower.py -> DAGGER_REPO_ID
  3. Fine-tune on the corrections in train_ur10_follower.py:
         DATASET_REPO_ID = "<DAGGER_REPO_ID>"   (or a merged originals+dagger set)
         PRETRAINED_PATH = "<base checkpoint dir>"
  4. Re-eval; repeat 2-3 against the new failure modes until satisfied.

Run (lerobot conda env; pendant payload set; e-stop in hand):
    python ur10_standalone_act/dagger_ur10_follower.py
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from relative_gamepad_teleop import RelativeGamepadTeleop, RelativeGamepadTeleopConfig
from ur10_follower import UR10Follower, UR10FollowerConfig

# Shared record/eval constants — IMPORTED (not duplicated) so cameras, crop boxes,
# EE/yaw bounds, reset-home pose AND the gamepad jog feel can NEVER drift from what
# produced the original dataset (a crop/bounds mismatch is a silent OOD failure).
from record_ur10_follower import (  # noqa: E402
    CONTROL_BACKEND,
    CAMERAS,
    CROP_BOXES,
    EE_BOUNDS_MIN,
    EE_BOUNDS_MAX,
    YAW_MIN,
    YAW_MAX,
    HOME_TCP,
    RESET_SETTLE_S,
    RANDOMIZATION_XY,
    RANDOMIZATION_Z,
    RANDOMIZATION_YAW,
    OPEN_GRIPPER_ON_RESET,
    EE_STEP,
    YAW_STEP,
    STICK_DEADZONE,
    STICK_CAL_S,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- user-tunable (dagger run params; geometry comes from the record import) -----
ROBOT_IP = "192.168.0.100"
MODEL_DIR = "outputs/act/ur10_follower/relative/last"   # base policy to correct
SRC_DATASET_REPO_ID = "local/ur10_follower_act_relative"  # for normalization stats + feature names
DAGGER_REPO_ID = "local/ur10_follower_act_relative_dagger"  # NEW correction dataset
DAGGER_TASK = "ur10 follower act relative"
NUM_EPISODES = 30
FPS = 30
EPISODE_TIME_S = 30      # safety upper bound; end early with TRIANGLE
RESET_TIME_S = 5         # reposition + grip window (gamepad, not recorded)

# HG-DAgger: keep an episode only if the operator actually corrected it (>=1 R1 frame).
# Uncorrected rollouts add nothing useful (success) or are harmful to train on
# (failure with no fix). Set False to keep every episode (matches eval_ur10_act.py).
SAVE_ONLY_CORRECTED_EPISODES = True

# Rerun logs EVERY frame with no throttle and is a prime cause of dropped frames at
# 30 Hz with 3 cameras + policy inference. OFF by default for full-rate collection.
USE_RERUN = False
USE_TTS = False
# -------------------------------------------------------------------------------


def _banner(msg: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def main() -> None:
    dt = 1.0 / FPS

    meta = LeRobotDatasetMetadata(SRC_DATASET_REPO_ID)
    policy: ACTPolicy = ACTPolicy.from_pretrained(MODEL_DIR)
    policy.eval()
    device = torch.device(policy.config.device)
    policy.to(device)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=MODEL_DIR, dataset_stats=meta.stats
    )
    logger.info("Policy from %s on %s (chunk=%d, n_action_steps=%d)",
                MODEL_DIR, device, policy.config.chunk_size, policy.config.n_action_steps)

    config = UR10FollowerConfig(
        id="ur10_follower", ip=ROBOT_IP, frequency=500,
        control_backend=CONTROL_BACKEND,
        kp_pos=5000.0, kp_rot=100.0, use_yaw=True, use_gripper=True, set_payload=False,
        payload_mass=1.3, cameras=CAMERAS, resolution=(224, 224), crop_boxes=CROP_BOXES,
        ee_bounds_min=EE_BOUNDS_MIN, ee_bounds_max=EE_BOUNDS_MAX,
        yaw_min=YAW_MIN, yaw_max=YAW_MAX,
        home_tcp=HOME_TCP, reset_settle_s=RESET_SETTLE_S,
        randomization_xy=RANDOMIZATION_XY, randomization_z=RANDOMIZATION_Z,
        randomization_yaw=RANDOMIZATION_YAW, open_gripper_on_reset=OPEN_GRIPPER_ON_RESET,
    )
    robot = UR10Follower(config)
    teleop = RelativeGamepadTeleop(RelativeGamepadTeleopConfig(
        id="ur10_rel_gamepad", ee_step=EE_STEP, yaw_step=YAW_STEP, deadzone=STICK_DEADZONE,
        use_yaw=True, use_gripper=True, stick_cal_s=STICK_CAL_S, fps=FPS,
        invert_delta_x=True, invert_delta_y=True, invert_delta_z=False, invert_delta_yaw=False,
    ))

    if USE_RERUN:
        from lerobot.utils.visualization_utils import init_rerun
        init_rerun(session_name="ur10_follower_dagger")

    episode_idx = 0
    robot_connected = teleop_connected = False
    dataset = None
    try:
        robot.connect()
        robot_connected = True
        teleop.connect()
        teleop_connected = True
        teleop.robot = robot          # rel-target clipping to ee_bounds/home
        events = teleop.events        # async, edge-detected gamepad events (same as record)
        rng = np.random.default_rng()

        # DAgger dataset: SAME schema as record_ur10_follower.py, straight from the
        # robot's own feature dicts so the fine-tune sees an identical layout.
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset = LeRobotDataset.create(
            repo_id=DAGGER_REPO_ID, fps=FPS, features={**action_features, **obs_features},
            robot_type=robot.name, use_videos=True, video_backend="libx264",
            image_writer_threads=4,
        )
        logger.info("DAgger dataset -> %s (corrections from policy %s)", DAGGER_REPO_ID, MODEL_DIR)

        def reset_window(duration_s):
            """Auto-home (+rand), then manual fine-position with R1 (same as eval)."""
            robot.go_to_home(rng)
            teleop.reset()
            t_end = time.perf_counter() + duration_s
            while time.perf_counter() < t_end and not events["exit_early"]:
                t0 = time.perf_counter()
                robot.send_action(teleop.get_action())
                precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            _banner(f"RESET (policy NOT running) — episode {episode_idx + 1}/{NUM_EPISODES}\n"
                    f"  HOLD R1 to reposition + grip. TRIANGLE starts the policy "
                    f"(auto-starts after {RESET_TIME_S}s).")
            log_say(f"Reset. Position the arm for episode {episode_idx + 1}", USE_TTS)
            reset_window(RESET_TIME_S)
            events["exit_early"] = False        # consume the 'start' press; keep stop/rerecord
            if events["stop_recording"]:        # CROSS during reset -> quit
                break

            robot.capture_home()      # anchor for the policy's relative outputs
            teleop.reset()
            policy.reset()
            events["exit_early"] = False
            events["rerecord_episode"] = False
            _banner(f"  ●  POLICY RUNNING (HG-DAgger) — episode {episode_idx + 1}/{NUM_EPISODES}\n"
                    f"  HOLD R1 to CORRECT (gamepad overrides + records the fix).\n"
                    f"  TRIANGLE=keep, SQUARE=redo, CROSS=stop (auto-ends after {EPISODE_TIME_S}s).")
            log_say(f"Episode {episode_idx + 1}", USE_TTS)

            step = 0
            n_intervention = 0
            episode_start = time.perf_counter()
            aborted = redo = False
            status = "TIMEOUT"
            while True:
                t0 = time.perf_counter()
                if not robot.controller_alive or robot.safety_status:
                    _banner("SAFETY STOP / controller dead — discarding episode + ending session")
                    aborted = True
                    break

                obs = robot.get_observation()

                # Policy proposes a relative-to-home action.
                obs_frame = build_inference_frame(
                    observation=obs, device=device, ds_features=meta.features
                )
                obs_t = preprocess(obs_frame)
                with torch.no_grad():
                    action = policy.select_action(obs_t)
                action = postprocess(action)
                policy_action = make_robot_action(action, meta.features)  # relative dict

                # HG-DAgger gate: R1 held -> human corrects; else policy drives and the
                # teleop shadows it for a bumpless handoff.
                if teleop.should_intervene():
                    action_dict = teleop.get_action()
                    n_intervention += 1
                else:
                    action_dict = policy_action
                    teleop.sync_to(policy_action)

                robot.send_action(action_dict)  # home added inside

                # Record the action that ACTUALLY drove the robot, paired with the
                # on-policy observation — the HG-DAgger training signal.
                obs_dframe = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                act_dframe = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
                dataset.add_frame({**obs_dframe, **act_dframe, "task": DAGGER_TASK})
                step += 1

                if USE_RERUN:
                    from lerobot.utils.visualization_utils import log_rerun_data
                    log_rerun_data(observation=obs, action=action_dict)

                # Async edge-detected episode buttons (same contract as record).
                if events["stop_recording"]:
                    status = "STOP"
                    break
                if events["rerecord_episode"]:
                    status = "REDO"
                    redo = True
                    break
                if events["exit_early"]:
                    status = "KEEP"
                    break
                if step >= int(EPISODE_TIME_S * FPS):
                    status = "TIMEOUT"
                    break
                precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

            wall = time.perf_counter() - episode_start
            print(f"  ■  episode {episode_idx + 1}/{NUM_EPISODES} {status} "
                  f"({step} frames, {wall:.1f}s, {step / max(wall, 1e-6):.1f} Hz, "
                  f"{n_intervention} corrected frames)", flush=True)

            # Discard on safety/redo, or when no correction was made (if gated).
            if aborted:
                dataset.clear_episode_buffer()
                break
            if status == "STOP":
                dataset.clear_episode_buffer()
                break
            if redo:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            if SAVE_ONLY_CORRECTED_EPISODES and n_intervention == 0:
                print("  ↳ no R1 correction — discarding (set SAVE_ONLY_CORRECTED_EPISODES=False "
                      "to keep).", flush=True)
                dataset.clear_episode_buffer()
                episode_idx += 1
                continue

            dataset.save_episode()
            print(f"  ✓  saved correction episode {episode_idx + 1}/{NUM_EPISODES} "
                  f"({step} frames, {n_intervention} corrected)", flush=True)
            episode_idx += 1

    except KeyboardInterrupt:
        logger.info("DAgger collection stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("DAgger collection failed")
    finally:
        if dataset is not None:
            try:
                dataset.finalize()
                logger.info("DAgger dataset finalized -> %s (%d episodes)", DAGGER_REPO_ID, episode_idx)
            except Exception:
                logger.exception("dataset.finalize failed")
        if robot_connected:
            try:
                robot.disconnect()
            except Exception:
                logger.exception("robot.disconnect failed")
        if teleop_connected:
            try:
                teleop.disconnect()
            except Exception:
                logger.exception("teleop.disconnect failed")


if __name__ == "__main__":
    main()
