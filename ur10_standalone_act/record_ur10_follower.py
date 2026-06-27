"""Record UR10e ACT demonstrations — standalone, RELATIVE action space, 30 Hz.

Structured 1-to-1 after the proven `act_train/record_dataset_ps4_joystick.py`:
the STANDARD lerobot `record_loop` + `make_default_processors`, a flat per-episode
while loop, and an ASYNC listener that drives `record_loop`'s `events` dict.

The only departure from that reference is WHERE `events` comes from: the reference
uses `init_keyboard_listener()`; we keep episode control ON THE GAMEPAD via a
background `GamepadListener` thread (same async contract, immune to loop rate — this
is what fixes the old "press success twice" / "1-frame episode" bugs).

UR10-specific features folded in (all live in UR10Follower / RelativeGamepadTeleop):
  - RELATIVE actions: per-episode capture_home(); obs/action are relative to it.
  - Grasp-at-start: go_to_home() opens the gripper; operator re-grips in the reset window.
  - EE bounds + yaw clip: enforced in send_action AND in the recorded action.
  - Programmatic reset home (+ optional randomization) via the OSC target.

Recorded action (5-D):   [x.pos, y.pos, z.pos, yaw.pos, gripper.pos]  (rel-to-home)
Recorded obs.state (11-D): [joint_pos(6), tcp_xyz_rel(3), yaw_offset(1), gripper(1)]
Images: front/side/wrist, cropped (hardcoded boxes) then resized to 224x224.

Gamepad: HOLD R1 to drive. Triangle = end episode + save. Square = re-record.
Cross = stop the whole session. Per episode: auto-home + grip window, then record.

Run (lerobot conda env; pendant payload set; e-stop in hand):
    python ur10_standalone_act/record_ur10_follower.py
"""

from __future__ import annotations

import logging
import time

import numpy as np

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from relative_gamepad_teleop import RelativeGamepadTeleop, RelativeGamepadTeleopConfig
from ur10_follower import UR10Follower, UR10FollowerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- user-tunable ---------------------------------------------------------------
ROBOT_IP = "192.168.0.100"
# Control backend: "servol" (UR's stiff position controller) for free-space pick/place/
# transport stages; "osc" (compliant task-space torque) for the contact-rich press stage.
# Datasets/ACT training are identical across backends — record each stage with the backend
# you will run it with at eval time.
CONTROL_BACKEND = "servol"
REPO_ID = "local/ur10_follower_act_relative"
TASK_DESCRIPTION = "ur10 follower act relative"
NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_S = 20          # record-window cap; end early with Triangle
RESET_TIME_S = 5            # reposition + grip window (gamepad, not recorded)
# If the episode hits EPISODE_TIME_S without a TRIANGLE save, the demo is treated as
# incomplete/truncated and re-done (discarded, same slot retried) instead of saved.
# Set False to save timed-out episodes. A button press (save/redo/discard) always wins.
REDO_ON_TIMEOUT = True

EE_STEP = (0.001, 0.001, 0.001)  # m per unit-action (R1 held)
YAW_STEP = 0.006                 # rad per unit-action
STICK_DEADZONE = 0.05
STICK_CAL_S = 1.5

# 3 RealSense by serial (front/side/wrist).
CAMERAS = {
    "front": RealSenseCameraConfig(serial_number_or_name="409122272284", fps=FPS, width=640, height=480),
    "side":  RealSenseCameraConfig(serial_number_or_name="409122273078", fps=FPS, width=640, height=480),
    "wrist": RealSenseCameraConfig(serial_number_or_name="323622272232", fps=FPS, width=640, height=480),
}
# Hardcoded crop boxes (top, left, height, width) applied before resize-to-224.
CROP_BOXES = {
    "front": (0, 176, 295, 336),
    "side":  (0, 153, 339, 421),
    "wrist": (47, 226, 343, 388),
}

# ABSOLUTE base-frame workspace clip (safety). Set via find_limits.py.
EE_BOUNDS_MIN = (-0.269, -0.586, 0.305)
EE_BOUNDS_MAX = (-0.227, -0.529, 0.352)
YAW_MIN, YAW_MAX = -0.4, 0.4

# Programmatic reset home: absolute TCP [x, y, z, rx, ry, rz] (OSC target, no moveJ).
HOME_TCP = [-0.252, -0.563, 0.352, 3.14159, 0.0, 0.0]
RESET_SETTLE_S = 5.0
RANDOMIZATION_XY = 0.0
RANDOMIZATION_Z = 0.0
RANDOMIZATION_YAW = 0.0
OPEN_GRIPPER_ON_RESET = True
USE_TTS = False   # no speaker on this PC -> rely on the on-screen banners below
# record_loop's display_data logs Rerun EVERY frame (no throttle) -> can drop the loop
# rate. Default OFF for fast/full-rate recording; review later with
# inspect_ur10_follower_dataset.py --rerun. Set True if you want live cropped images.
# (Was the prime cause of dropped frames at 30 Hz with 3 cameras — keep OFF to record
# at full rate; review images afterwards with inspect_ur10_follower_dataset.py --rerun.)
USE_RERUN = False
# -------------------------------------------------------------------------------


def _banner(msg: str) -> None:
    """Loud, always-visible console marker (stdout). The reset vs. record phases used to
    be distinguishable only by log_say (audio); with no speaker the operator could not
    tell when the (non-recorded) reset window ended and the recorded window began —
    pressing Triangle in the reset window then again at record start gave tiny episodes."""
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def main() -> None:
    robot = UR10Follower(UR10FollowerConfig(
        id="ur10_follower", ip=ROBOT_IP, frequency=500,
        control_backend=CONTROL_BACKEND,
        kp_pos=5000.0, kp_rot=100.0, use_yaw=True, use_gripper=True,
        set_payload=False, payload_mass=1.3,
        cameras=CAMERAS, resolution=(224, 224), crop_boxes=CROP_BOXES,
        ee_bounds_min=EE_BOUNDS_MIN, ee_bounds_max=EE_BOUNDS_MAX,
        yaw_min=YAW_MIN, yaw_max=YAW_MAX,
        home_tcp=HOME_TCP, reset_settle_s=RESET_SETTLE_S,
        randomization_xy=RANDOMIZATION_XY, randomization_z=RANDOMIZATION_Z,
        randomization_yaw=RANDOMIZATION_YAW, open_gripper_on_reset=OPEN_GRIPPER_ON_RESET,
    ))
    teleop = RelativeGamepadTeleop(RelativeGamepadTeleopConfig(
        id="ur10_rel_gamepad", ee_step=EE_STEP, yaw_step=YAW_STEP, deadzone=STICK_DEADZONE,
        use_yaw=True, use_gripper=True, stick_cal_s=STICK_CAL_S, fps=FPS,
        invert_delta_x=True, invert_delta_y=True, invert_delta_z=False, invert_delta_yaw=False,
    ))

    # Everything that allocates a hardware resource (OSC controller process, cameras,
    # gamepad thread) is INSIDE try/finally so any failure here — e.g. the dataset dir
    # already existing — still tears the robot/teleop down instead of orphaning the OSC
    # controller process.
    rng = np.random.default_rng()
    episode_idx = 0
    robot_connected = teleop_connected = False
    dataset = None
    try:
        # Connect robot + teleop. The teleop's background GamepadListener owns the `events`
        # dict (Triangle->exit_early, Square->rerecord, Cross->stop) — set asynchronously,
        # exactly like init_keyboard_listener() in the reference script.
        robot.connect()
        robot_connected = True
        teleop.connect()
        teleop_connected = True
        teleop.robot = robot              # for relative-target clipping to ee_bounds/home
        events = teleop.events            # gamepad episode control -> record_loop

        # Configure the dataset features (relative-action robot).
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset_features = {**action_features, **obs_features}

        dataset = LeRobotDataset.create(
            repo_id=REPO_ID, fps=FPS, features=dataset_features,
            robot_type=robot.name, use_videos=True, video_backend="libx264",
            image_writer_threads=4,
        )

        teleop_action_processor, robot_action_processor, robot_observation_processor = \
            make_default_processors()
        if USE_RERUN:
            init_rerun(session_name="ur10_follower_record")

        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            # --- reset: auto-home + manual reposition/grip (gamepad, NOT recorded) ---
            _banner(f"RESET (NOT recording) — episode {episode_idx + 1}/{NUM_EPISODES}\n"
                    f"  HOLD R1 to move. Reposition + grip the object.\n"
                    f"  Press TRIANGLE when ready to START recording "
                    f"(auto-starts after {RESET_TIME_S}s).")
            log_say(f"Reset. Episode {episode_idx + 1}", USE_TTS)
            robot.go_to_home(rng)
            teleop.reset()
            record_loop(
                robot=robot, events=events, fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop, control_time_s=RESET_TIME_S,
                # display_data=False here on purpose: the reset window saves nothing, but
                # record_loop still streams to Rerun whenever display_data is true. Showing
                # it made Rerun "come alive" during reset — looking like recording had
                # started ~RESET_TIME_S before the "RECORDING NOW" banner. Rerun now lights
                # up only for the real record_loop below, in sync with the banner.
                single_task=TASK_DESCRIPTION, display_data=False,
            )
            events["exit_early"] = False
            events["episode_failed"] = False
            if events["stop_recording"]:
                break

            # --- record window ---
            robot.capture_home()   # anchor: rel actions/obs relative to THIS pose
            teleop.reset()         # rel starts at 0 from the new home (also resets n_calls)
            _banner(f"  ●  RECORDING NOW — episode {episode_idx + 1}/{NUM_EPISODES}\n"
                    f"  Do the task. TRIANGLE=save, CIRCLE=discard, SQUARE=redo, CROSS=stop.\n"
                    f"  (auto-saves after {EPISODE_TIME_S}s)")
            log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}", USE_TTS)
            _t0 = time.perf_counter()
            record_loop(
                robot=robot, events=events, fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop, dataset=dataset, control_time_s=EPISODE_TIME_S,
                single_task=TASK_DESCRIPTION, display_data=USE_RERUN,
            )
            _wall = time.perf_counter() - _t0
            # print() not logger.info(): a lerobot import pre-configures the root logger, so
            # logging.basicConfig(INFO) above is a no-op and INFO lines are suppressed. We
            # always want this rate diagnostic visible.
            _ended = "button" if _wall < EPISODE_TIME_S - 0.5 else "timeout"
            print(f"  ■  RECORDING STOPPED — {teleop.n_calls} frames in {_wall:.2f}s = "
                  f"{teleop.n_calls / max(_wall, 1e-6):.1f} Hz "
                  f"(target {FPS}, cap {EPISODE_TIME_S}s) ended_by={_ended}", flush=True)
            if _ended == "button" and _wall < 2.0:
                print("  !! very short episode — if unintended, this was likely a stray "
                      "TRIANGLE press; use SQUARE next time to re-record.", flush=True)

            # --- safety: discard the in-progress episode on protective/e-stop or dead controller ---
            if not robot.controller_alive or robot.safety_status:
                _banner("SAFETY STOP — discarding episode and stopping session")
                log_say("Safety stop — discarding episode and stopping", USE_TTS)
                dataset.clear_episode_buffer()
                break

            if events["rerecord_episode"]:
                _banner("RE-RECORD requested (SQUARE) — discarding this episode")
                log_say("Re-recording", USE_TTS)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if events["episode_failed"]:
                _banner("FAILURE (CIRCLE) — discarding this take")
                events["episode_failed"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if REDO_ON_TIMEOUT and _ended == "timeout":
                _banner(f"TIMEOUT ({EPISODE_TIME_S}s, no TRIANGLE) — incomplete demo, "
                        "discarding + redoing this episode")
                log_say("Timeout — redoing episode", USE_TTS)
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            print(f"  ✓  saved episode {episode_idx + 1}/{NUM_EPISODES} "
                  f"({teleop.n_calls} frames)", flush=True)
            episode_idx += 1

    except KeyboardInterrupt:
        logger.info("Recording stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("Recording failed")
    finally:
        log_say("Stop recording", USE_TTS)
        if dataset is not None:
            try:
                dataset.finalize()
                logger.info("Dataset finalized -> %s (%d episodes)", REPO_ID, episode_idx)
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
