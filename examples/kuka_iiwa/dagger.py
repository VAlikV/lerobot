"""
Policy rollout + operator corrections

Keyboard Controls:
    p   - pause / resume the policy
    c   - start / finish a correction (start: rewind prompt -> sync prompt -> recording;
          finish: the correction is moved to the pending buffer as one episode)
    r   - discard the correction currently being recorded (and go back to PAUSED)
    s   - toggle action scaling (same as in record script)
    esc - stop the session (a correction in progress is kept)
"""

import argparse
import enum
import json
import threading
import time
from pathlib import Path
from collections import deque
from threading import Event, Lock

import draccus
import torch

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.kuka_iiwa import KukaIiwa
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import (
    GripperPositionToDiscrete,
    KukaJointBoundsAndSafety,
    KukaJointDeltaScaling,
)
from lerobot.teleoperators.kuka_leader import KukaLeader
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from record_config import RecordingConfig
from record_utils import _assert_pending_episode_indices, _finish_episode_buffer, leader_gripper_sync_position

# Keys
KEY_PAUSE_RESUME = "p"
KEY_CORRECTION = "c"
KEY_DISCARD = "r"
KEY_SCALE = "s"
KEY_STOP = "esc"


# ---------------------------------------------------------------------------
# State machine + thread-safe events
# ---------------------------------------------------------------------------


class DAggerPhase(enum.Enum):
    AUTONOMOUS = "autonomous"  # policy drives the follower
    PAUSED = "paused"  # follower holds the last action
    CORRECTING = "correcting"  # operator drives via leader, frames are recorded


_TRANSITIONS: dict[tuple[DAggerPhase, str], DAggerPhase] = {
    (DAggerPhase.AUTONOMOUS, "pause_resume"): DAggerPhase.PAUSED,
    (DAggerPhase.PAUSED, "pause_resume"): DAggerPhase.AUTONOMOUS,
    (DAggerPhase.PAUSED, "correction"): DAggerPhase.CORRECTING,
    (DAggerPhase.CORRECTING, "correction"): DAggerPhase.PAUSED,
}


class DAggerEvents:
    """Keyboard thread writes transition requests, the main loop consumes them."""

    def __init__(self, initial_phase: DAggerPhase = DAggerPhase.PAUSED) -> None:
        self._lock = Lock()
        self._phase = initial_phase
        self._pending: str | None = None
        self._apply_scale = False
        self.stop_recording = Event()
        self.discard_correction = Event()

    @property
    def phase(self) -> DAggerPhase:
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: DAggerPhase) -> None:
        with self._lock:
            self._phase = value

    @property
    def apply_scale(self) -> bool:
        with self._lock:
            return self._apply_scale

    @apply_scale.setter
    def apply_scale(self, value: bool) -> None:
        with self._lock:
            self._apply_scale = value

    def toggle_scale(self) -> None:
        with self._lock:
            self._apply_scale = not self._apply_scale
            print(f"[DAgger] action scaling: {self._apply_scale}", flush=True)

    def request_transition(self, event: str) -> None:
        with self._lock:
            if (self._phase, event) in _TRANSITIONS:
                self._pending = event

    def request_discard(self) -> None:
        """Finish the current correction and throw its frames away."""
        with self._lock:
            if self._phase == DAggerPhase.CORRECTING:
                self.discard_correction.set()
                self._pending = "correction"

    def consume_transition(self) -> tuple[DAggerPhase, DAggerPhase] | None:
        with self._lock:
            if self._pending is None:
                return None
            key = (self._phase, self._pending)
            self._pending = None
            new_phase = _TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def clear_pending(self) -> None:
        """Drop key presses made while the operator was typing into a blocking prompt."""
        with self._lock:
            self._pending = None

    def reset(self) -> None:
        """Reset all transient state for a fresh session."""
        with self._lock:
            self._phase = DAggerPhase.AUTONOMOUS
            self._pending_transition = None
        self.upload_requested.clear()


def _init_keyboard(events: DAggerEvents):
    def dispatch(name: str) -> None:
        name = name.lower()
        if name == KEY_STOP:
            print("[DAgger] stop requested", flush=True)
            events.stop_recording.set()
        elif name == KEY_PAUSE_RESUME:
            events.request_transition("pause_resume")
        elif name == KEY_CORRECTION:
            events.request_transition("correction")
        elif name == KEY_DISCARD:
            events.request_discard()
        elif name == KEY_SCALE:
            events.toggle_scale()

    return create_key_listener(
        dispatch,
        controls_help=(
            f"pause/resume='{KEY_PAUSE_RESUME}', correction='{KEY_CORRECTION}', "
            f"discard='{KEY_DISCARD}', scale='{KEY_SCALE}', {KEY_STOP}=stop"
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> RecordingConfig:
    with open(path, "r") as f:
        raw_cfg = json.load(f)
    return draccus.decode(RecordingConfig, raw_cfg)


def _say(text: str, play_sound: bool) -> None:
    print(f"[TTS] {text}", flush=True)
    log_say(text, play_sound)


def _ask_rewind_seconds(available_s: float) -> float:
    raw = input(
        f"\nRewind follower along executed trajectory? "
        f"seconds back [0..{available_s:.1f}] (Enter = no rewind): "
    ).strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, min(float(raw.replace(",", ".")), available_s))
    except ValueError:
        print(f"Could not parse '{raw}', no rewind.")
        return 0.0


# ---------------------------------------------------------------------------
# DAgger session
# ---------------------------------------------------------------------------
 
 
class KukaDAgger:
    def __init__(
        self,
        cfg: RecordingConfig,
        follower: KukaIiwa,
        leader: KukaLeader,
        pipeline: RobotProcessorPipeline,
        safety_pipeline: RobotProcessorPipeline,
        joint_scaler: KukaJointDeltaScaling,
        dataset: LeRobotDataset,
        policy: ACTPolicy,
        preprocess,
        postprocess,
        policy_features: dict,
        events: DAggerEvents,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.fps = cfg.fps
        self.follower = follower
        self.leader = leader
        self.pipeline = pipeline
        self.safety_pipeline = safety_pipeline
        self.joint_scaler = joint_scaler
        self.dataset = dataset
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.policy_features = policy_features
        self.events = events
        self.device = device
 
        self.use_relative_actions = policy.config.use_relative_actions
        self.action_queue: deque = deque()
        self.history: deque = deque(maxlen=max(1, int(cfg.dagger.max_rewind_s * self.fps)))
        self.last_action: dict | None = None
 
        self.pending_episode_buffers: list[dict] = []
        self.max_corrections = cfg.dataset.num_episodes
        self.use_tts = cfg.dataset.use_tts
 
    # -- AUTONOMOUS --------------------------------------------------------
 
    def _resume_policy(self) -> None:
        self.policy.reset()
        for proc in (self.preprocess, self.postprocess):
            if hasattr(proc, "reset"):
                proc.reset()
        self.action_queue.clear()
        self.history.clear()
 
    def _autonomous_step(self) -> None:
        if not self.use_relative_actions:
            obs = self.follower.get_observation()
            frame = build_inference_frame(
                observation=obs, ds_features=self.policy_features, device=self.device
            )
            observation = self.preprocess(frame)
            with torch.no_grad():
                action = self.policy.select_action(observation)
            action = self.postprocess(action)
        elif self.policy.config.temporal_ensemble_coeff is not None:
            obs = self.follower.get_observation()
            frame = build_inference_frame(
                observation=obs, ds_features=self.policy_features, device=self.device
            )
            observation = self.preprocess(frame)
            with torch.no_grad():
                chunk = self.policy.predict_action_chunk(observation)
            # Relative chunks must share absolute coordinates before averaging.
            chunk = self.postprocess(chunk)
            action = self.policy.temporal_ensembler.update(chunk)
        else:
            if len(self.action_queue) == 0:
                obs = self.follower.get_observation()
                frame = build_inference_frame(
                    observation=obs, ds_features=self.policy_features, device=self.device
                )
                observation = self.preprocess(frame)
                with torch.no_grad():
                    chunk = self.policy.predict_action_chunk(observation)
                chunk = chunk[:, :self.cfg.dagger.n_action_steps]
                chunk = self.postprocess(chunk)
                self.action_queue.extend(chunk.squeeze(0))
            action = self.action_queue.popleft().unsqueeze(0)
 
        robot_action = make_robot_action(action, self.policy_features)
        robot_action = self.safety_pipeline(robot_action)
        sent = self.follower.send_action(robot_action)
 
        self.last_action = sent
        self.history.append(sent)  # trajectory for rewind
 
    # -- PAUSED -> CORRECTING ----------------------------------------------
 
    def _rewind(self, seconds: float) -> None:
        """Reverse playback of the last `seconds` of executed (sent) actions."""
        n = min(int(round(seconds * self.fps)), len(self.history))
        if n <= 0:
            return
 
        path = list(self.history)[-n:][::-1]  # newest -> oldest
        dt = 1.0 / (self.fps * self.cfg.dagger.rewind_speed)
        print(f"[DAgger] rewinding {n / self.fps:.1f}s ({n} steps)...", flush=True)
 
        done = 0
        for action in path:
            if self.events.stop_recording.is_set():
                break
            t0 = time.perf_counter()
            self.last_action = self.follower.send_action(action)
            done += 1
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
 
        for _ in range(done):
            self.history.pop()
 
    def _enter_correction(self) -> None:
 
        # 1) optional rewind
        available_s = len(self.history) / self.fps
        if available_s >= 1.0 / self.fps:
            rewind_s = _ask_rewind_seconds(available_s)
            if rewind_s > 0:
                self._rewind(rewind_s)
 
        # 2) confirmation for leader sync
        input("\nPress Enter to sync leader to follower position...")
 
        # 3) drive leader to the follower pose
        follower_obs = self.follower.get_observation()
        follower_obs["gripper.pos"] = leader_gripper_sync_position(
            follower_obs["gripper.pos"], self.cfg.pipeline.gripper_threshold, self.cfg.pipeline.gripper_reverse
        )
        self.leader.send_goal_position(
            follower_obs,
            hold_s=3.0,
            tolerance_counts=100,
        )
 
        # 4) fresh teleop state
        self.joint_scaler.reset()
        self.events.apply_scale = False
        self.leader.reset_filter()
 
        self.history.clear()  # the old trajectory is no longer contiguous
        self.events.discard_correction.clear()
        self.events.clear_pending()
        print("[DAgger] CORRECTING: recording. Press 'c' to finish, 'r' to discard.", flush=True)
 
    # -- CORRECTING --------------------------------------------------------
 
    def _correction_step(self) -> None:
        obs = self.follower.get_observation()
        self.joint_scaler.set_enabled(self.events.apply_scale)
 
        leader_action = self.leader.get_action()
        leader_action = self.pipeline((leader_action, obs))
        sent_action = self.follower.send_action(leader_action)
        self.last_action = sent_action
 
        observation_frame = build_dataset_frame(self.dataset.features, obs, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.dataset.features, sent_action, prefix=ACTION)
        self.dataset.add_frame(
            {**observation_frame, **action_frame, "task": self.cfg.dataset.task}
        )
 
    def _exit_correction(self, discard: bool) -> None:
        if discard:
            _finish_episode_buffer(self.dataset, rerecord=True)
            print("[DAgger] correction discarded", flush=True)
            _say("Correction discarded", self.use_tts)
        elif self.dataset.has_pending_frames():
            self.pending_episode_buffers.append(_finish_episode_buffer(self.dataset, rerecord=False))
            n = len(self.pending_episode_buffers)
            print(f"[DAgger] correction {n}/{self.max_corrections} buffered", flush=True)
            _say(f"Correction {n} recorded", self.use_tts)
        else:
            print("[DAgger] empty correction, nothing to save", flush=True)
        self.events.discard_correction.clear()
 
    # -- main loop ---------------------------------------------------------
 
    def _handle_transition(self, old: DAggerPhase, new: DAggerPhase) -> None:
        print(f"[DAgger] {old.value} -> {new.value}", flush=True)
 
        if old == DAggerPhase.PAUSED and new == DAggerPhase.CORRECTING:
            self._enter_correction()
        elif old == DAggerPhase.CORRECTING and new == DAggerPhase.PAUSED:
            self._exit_correction(discard=self.events.discard_correction.is_set())
        elif new == DAggerPhase.AUTONOMOUS:
            self._resume_policy()
        # else AUTONOMOUS -> PAUSED
 
    def run(self) -> None:
        events = self.events
        dt = 1.0 / self.fps
        print(f"[DAgger] ready. Phase: {events.phase.value}. Press '{KEY_PAUSE_RESUME}' to start the policy.")
 
        while (
            len(self.pending_episode_buffers) < self.max_corrections
            and not events.stop_recording.is_set()
        ):
            t0 = time.perf_counter()
 
            transition = events.consume_transition()
            if transition is not None:
                self._handle_transition(*transition)
                t0 = time.perf_counter()
 
            phase = events.phase
            if phase == DAggerPhase.AUTONOMOUS:
                self._autonomous_step()
            elif phase == DAggerPhase.CORRECTING:
                self._correction_step()
 
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
 
        # stopped in the middle of a correction -> keep what has been recorded
        if events.phase == DAggerPhase.CORRECTING:
            self._exit_correction(discard=events.discard_correction.is_set())
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Run KUKA DAgger and record operator corrections.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration.")
    parser.add_argument("--dataset-id", required=True, help="Correction dataset repo ID to create or resume.")
    parser.add_argument(
        "--policy-dataset-id", required=True, help="Source dataset repo ID for policy features and statistics."
    )
    parser.add_argument("--model-id", required=True, help="Model repo ID or local checkpoint path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.dataset.repo_id = args.dataset_id
    fps = cfg.fps
    device = torch.device(cfg.dagger.device if torch.cuda.is_available() or cfg.dagger.device == "cpu" else "cpu")
 
    follower = KukaIiwa(cfg.robot)
    leader = KukaLeader(cfg.leader)
 
    # teleop pipeline
    joint_scaler = KukaJointDeltaScaling(scale_factor=cfg.pipeline.scale_factor)
    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            joint_scaler,
            KukaJointBoundsAndSafety(joint_offset_deg=cfg.pipeline.joint_offset_deg),
            GripperPositionToDiscrete(
                threshold=cfg.pipeline.gripper_threshold,
                reverse=cfg.pipeline.gripper_reverse,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
 
    # policy-side safety
    safety_pipeline = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[KukaJointBoundsAndSafety(joint_offset_deg=cfg.pipeline.joint_offset_deg)],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )
 
    # policy
    policy_meta = LeRobotDatasetMetadata(args.policy_dataset_id)
    policy_config = ACTConfig.from_pretrained(args.model_id)
    policy_config.temporal_ensemble_coeff = cfg.dagger.temporal_ensemble_coeff
    policy_config.n_action_steps = 1 if cfg.dagger.temporal_ensemble_coeff is not None else cfg.dagger.n_action_steps
    policy = ACTPolicy.from_pretrained(args.model_id, config=policy_config)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model_id,
        dataset_stats=policy_meta.stats,
    )
 
    # dataset for corrections
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
 
    dataset_root = Path(cfg.dataset.root) if cfg.dataset.root else HF_LEROBOT_HOME / cfg.dataset.repo_id
    if dataset_root.exists():
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=dataset_root,
            image_writer_processes=0,
            image_writer_threads=4 * max(num_cameras, 1),
        )
        sanity_check_dataset_robot_compatibility(dataset, follower, fps, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            fps,
            root=dataset_root,
            robot_type=follower.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * max(num_cameras, 1),
            streaming_encoding=False,
        )
 
    follower.connect()
    leader.connect()
 
    events = DAggerEvents(initial_phase=DAggerPhase.PAUSED)
    listener = _init_keyboard(events)
 
    session = KukaDAgger(
        cfg=cfg,
        follower=follower,
        leader=leader,
        pipeline=pipeline,
        safety_pipeline=safety_pipeline,
        joint_scaler=joint_scaler,
        dataset=dataset,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        policy_features=policy_meta.features,
        events=events,
        device=device,
    )
 
    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")
 
        if cfg.dagger.reset_to_pose:
            current = follower.get_observation()
            target = dict(zip(follower.action_features.keys(), cfg.dataset.reset_pose))
            follower_smooth_move_to(follower, current, target, duration_s=3.0)
 
        session.run()
 
    except KeyboardInterrupt:
        print("\nStopping DAgger...")
 
    finally:
        if leader.is_connected:
            leader.disconnect()
        if follower.is_connected:
            follower.disconnect()
 
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
 
        buffers = session.pending_episode_buffers
        _assert_pending_episode_indices(dataset, buffers)
        for idx, episode_buffer in enumerate(buffers, start=1):
            print(f"Saving correction {idx}/{len(buffers)} after disconnect")
            dataset.save_episode(episode_data=episode_buffer)
 
        dataset.finalize()
 
        if listener is not None:
            listener.stop()
 
        print(f"Dataset finalized -> {cfg.dataset.repo_id}")
 
 
if __name__ == "__main__":
    main()

