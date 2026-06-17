"""Closed-loop ACT inference for the standalone UR10 follower (relative actions).

Mirrors act_train/act_using_example.py. The policy outputs RELATIVE targets; the
robot's send_action() adds the per-episode home internally, so there is NO manual
relative->absolute conversion here (unlike eval_ur10_act_v2_relative.py).

Per episode: reposition with the gamepad (R1) during a reset window, capture_home(),
then run the policy. Episode control uses the SAME robust async GamepadListener as the
record script (RelativeGamepadTeleop -> teleop.events), so button presses are never
missed at the loop rate: TRIANGLE=success, SQUARE=redo episode, CROSS=stop session.

    python ur10_standalone_act/eval_ur10_follower.py
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from relative_gamepad_teleop import RelativeGamepadTeleop, RelativeGamepadTeleopConfig
from ur10_follower import UR10Follower, UR10FollowerConfig

# Shared record/eval constants — IMPORTED (not duplicated) so cameras, crop boxes, EE/yaw
# bounds, reset-home pose AND the gamepad jog feel can NEVER drift from what produced the
# dataset. A crop or bounds mismatch is a silent eval failure (OOD images / clipped actions).
from record_ur10_follower import (  # noqa: E402
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

# -- user-tunable (eval-only run params; geometry comes from record import above) ----
ROBOT_IP = "192.168.0.100"
MODEL_DIR = "outputs/act/ur10_follower/relative/step_65000"
DATASET_REPO_ID = "local/ur10_follower_act_relative"   # for normalization stats + feature names
NUM_EPISODES = 20
FPS = 30
EPISODE_TIME_S = 20
RESET_TIME_S = 5
USE_TTS = False   # no speaker -> rely on the printed banners below
# -------------------------------------------------------------------------------


def _banner(msg: str) -> None:
    """Loud, always-visible console marker (logger.info is suppressed by a lerobot import;
    log_say is silent with no speaker)."""
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def main() -> None:
    dt = 1.0 / FPS

    meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
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

    episode_idx = 0
    robot_connected = teleop_connected = False
    try:
        robot.connect()
        robot_connected = True
        teleop.connect()
        teleop_connected = True
        teleop.robot = robot          # rel-target clipping to ee_bounds/home
        events = teleop.events         # async, edge-detected gamepad events (same as record)
        rng = np.random.default_rng()

        def reset_window(duration_s):
            """Auto-drive to home (+rand), then manual fine-position with R1 — using the SAME
            RelativeGamepadTeleop as record. Ends early when the operator presses an episode
            button (events['exit_early'])."""
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
            events["exit_early"] = False       # consume the 'start' press; keep stop/rerecord
            if events["stop_recording"]:       # CROSS during reset -> quit
                break

            robot.capture_home()      # anchor for the policy's relative outputs
            teleop.reset()
            policy.reset()
            events["exit_early"] = False
            events["rerecord_episode"] = False
            _banner(f"  ●  POLICY RUNNING — episode {episode_idx + 1}/{NUM_EPISODES}\n"
                    f"  TRIANGLE=success, SQUARE=redo, CROSS=stop session "
                    f"(auto-stops after {EPISODE_TIME_S}s).")
            log_say(f"Episode {episode_idx + 1}", USE_TTS)

            step = 0
            episode_start = time.perf_counter()
            aborted = redo = False
            status = "TIMEOUT"
            while True:
                t0 = time.perf_counter()
                if not robot.controller_alive or robot.safety_status:
                    _banner("SAFETY STOP / controller dead — ending episode + session")
                    aborted = True
                    break
                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs, device=device, ds_features=meta.features
                )
                obs_t = preprocess(obs_frame)
                with torch.no_grad():
                    action = policy.select_action(obs_t)
                action = postprocess(action)
                action_dict = make_robot_action(action, meta.features)  # relative target
                robot.send_action(action_dict)  # home added inside
                step += 1

                # Async edge-detected episode buttons (same contract as record):
                #   CROSS -> stop_recording, SQUARE -> rerecord_episode, TRIANGLE -> exit_early.
                if events["stop_recording"]:
                    status = "STOP"
                    break
                if events["rerecord_episode"]:
                    status = "REDO"
                    redo = True
                    break
                if events["exit_early"]:
                    status = "SUCCESS"
                    break
                if step >= int(EPISODE_TIME_S * FPS):
                    status = "TIMEOUT"
                    break
                precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

            print(f"  ■  episode {episode_idx + 1}/{NUM_EPISODES} {status} "
                  f"({step} steps, {time.perf_counter() - episode_start:.1f}s)", flush=True)

            if aborted or status == "STOP":
                break
            if redo:                  # SQUARE: re-do this episode, don't count it
                events["rerecord_episode"] = False
                events["exit_early"] = False
                continue
            episode_idx += 1

    except KeyboardInterrupt:
        logger.info("Inference stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("Inference failed")
    finally:
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
