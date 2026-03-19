"""
RC10 HIL-SERL training script
"""

import logging
import multiprocessing as mp
import signal
import time
from pathlib import Path
from queue import Empty, Full

import cv2
import numpy as np
import torch
import torch.nn.utils
import torch.optim as optim

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.robots.rc10_follower import RC10EnvConfig, RC10FollowerConfig, RC10FollowerCut, RC10RobotEnv
from lerobot.teleoperators.ps4_joystick import PS4JoystickTeleop, PS4JoystickTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logging.basicConfig(level=logging.INFO)
DEVICE = "cuda"
FPS = 30
MAX_EPISODES = 300
EPISODE_TIME_S = 30.0
RESET_TIME_S = 7.0
MAX_EPISODE_STEPS = int(EPISODE_TIME_S * FPS)
BATCH_SIZE = 64     # Reduced from 256 to fit in 5.65 GB GPU. Use 256 with larger GPUs.
ONLINE_BUFFER_CAPACITY = 100000 # At 30 FPS and with max 30 second episodes , each episode is 30 fps x 30 seconds = 900 transitions per episode, so 100k can hold 100 episodes easily
OFFLINE_BUFFER_CAPACITY = 100000  # Same sizing logic, preloaded with our 20 recordered episodes demos which is approx 30 fps x 30 seconds x 20 episodes = 18000 transitions. It grows as we intervene during training
ONLINE_STEPS_BEFORE_LEARNING = 200 # Wait this many transitions before first training step
LEARNING_RATE = 3e-4    # Adam optimizer learning rate for critic , actor and temperature. standard value for SAC
GRAD_CLIP_NORM = 40.0   # Max gradient norm before clipping (prevents gradient explosions) default value for hil-serl
POLICY_UPDATE_FREQ = 2  # Update actor and temperature every N critic updates. since critic learns faster than actor (updating actor less frequently stabilizes training)
SEND_PARAMS_EVERY = 50   # Send updated policy weights to actor every N training steps.
SAVE_EVERY = 500    # Save checkpoint every N training steps.
OUTPUT_DIR = Path("outputs/rc10_hilserl")
IMAGE_SIZE = (64, 64) # 64x64 to fit in 5.65 GB GPU. Use (128, 128) with larger GPUs.


# We need to get these crop parameters using src/lerobot/rl/crop_dataset_roi.py script
IMAGE_CROP = {
    # "front": (51, 3, 71, 121),
    # "side": (top, left, height, width),
    # "gripper": (top, left, height, width),
}

DEMO_DATASET_REPO_ID = "local/rc10_hilserl_demos"  # or None

robot_config = RC10FollowerConfig(
    id="my_rc10_follower",
    ip="10.10.10.10",
    rate_hz=100,
    velocity=1.0,
    acceleration=1.0,
    threshold_position=0.001,
    threshold_angle=1.0,
    gripper_port="/dev/ttyUSB0",
    gripper_baudrate=115200,
    cameras={
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "side": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
        "gripper": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),

    },
    resolution=(128,128),
)

teleop_config = PS4JoystickTeleopConfig(
    id="my_teleop_ps4_joystick",
    max_speed=0.05,
    max_rot_speed=0.5,
    deadzone=0.05,
    alpha=0.3,
    poll_rate=100,
    x_init=0.095,
    y_init=0.35,
    z_init=0.23,
    roll_init=np.pi,
    pitch_init=0.0,
    yaw_init=0.0,
    delta_mode=True,  # normalized [-1, 1] deltas
)

env_config = RC10EnvConfig(
    ee_step_sizes={"x": 0.002, "y": 0.002, "z": 0.002, "yaw": 0.02},
    ee_bounds={
        "min": [-0.0771, 0.2554, 0.2296],   #obtain these using the rc10/rc10_find_ee_limits.py file
        "max": [0.2836, 0.6417, 0.4079],
    },
    fixed_roll=np.pi,
    fixed_pitch=0.0,
    reset_tcp= [0.095, 0.35, 0.23, np.pi, 0.0, 0.0],    # home pose
    reset_time_s=RESET_TIME_S,
    display_cameras=False,
    use_gripper=True,
)

policy_config = SACConfig(
    device=DEVICE,
    storage_device=DEVICE,
    vision_encoder_name="helper2424/resnet10",
    freeze_vision_encoder=False,
    shared_encoder=True,
    image_encoder_hidden_dim=32,
    image_embedding_pooling_dim=8,
    state_encoder_hidden_dim=256,
    latent_dim=256,
    num_critics=2,
    num_discrete_actions=None,
    discount=0.99,
    temperature_init=0.01,
    critic_lr=LEARNING_RATE,
    actor_lr=LEARNING_RATE,
    temperature_lr=LEARNING_RATE,
    critic_target_update_weight=0.005,
    utd_ratio=1,
    grad_clip_norm=GRAD_CLIP_NORM,
    online_steps=100000,
    online_buffer_capacity=ONLINE_BUFFER_CAPACITY,
    offline_buffer_capacity=OFFLINE_BUFFER_CAPACITY,
    online_step_before_learning=ONLINE_STEPS_BEFORE_LEARNING,
    use_torch_compile=False,
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMAGE_SIZE)),
        f"{OBS_IMAGES}.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMAGE_SIZE)),
        f"{OBS_IMAGES}.gripper": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMAGE_SIZE)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
    },
    dataset_stats={
        f"{OBS_IMAGES}.front": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        f"{OBS_IMAGES}.side": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        f"{OBS_IMAGES}.gripper": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        OBS_STATE: {"min": [-0.0771, 0.2554, 0.2296, -3.15, -1.0], "max": [0.2836, 0.6417, 0.4079, 3.15, 1.0]},
        ACTION: {"min": [-1.0]*5, "max": [1.0]*5},
    },
)


def obs_to_policy_input(obs: dict, device: str = "cpu") -> dict:
    """Convert raw env observation to policy input tensors
    Applies crop (if crop params are set) -> resize -> normalize -> to device"""
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device=device)

    images = {}
    for cam_name, image in obs["pixels"].items():
        if cam_name in IMAGE_CROP:
            top, left, height, width = IMAGE_CROP[cam_name]
            image = image[top:top + height, left:left + width]
        # Resize + uint8 HWC -> float32 CHW normalized to [0, 1]
        image = cv2.resize(image, IMAGE_SIZE)
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        images[f"{OBS_IMAGES}.{cam_name}"] = image.to(device=device)

    return {OBS_STATE: state, **images}


# Actor

def run_actor(
        transitions_queue: mp.Queue,
        parameters_queue: mp.Queue,
        shutdown_event: mp.Event,
):
    """Collect experience by running the policy in the env
    We can intervene via ps4 joysticl (hold R1)"""

    robot = RC10FollowerCut(robot_config)
    teleop = PS4JoystickTeleop(teleop_config)
    teleop.connect()

    env = RC10RobotEnv(robot=robot, env_config=env_config)

    policy = SACPolicy(policy_config)
    actor_device = "cpu"  # Actor runs on CPU to save GPU memory for the learner
    policy.eval()
    policy.to(device=actor_device)

    dt = 1.0 / FPS

    try:
        for episode in range(MAX_EPISODES):
            if shutdown_event.is_set():
                break

            obs, _info = env.reset()
            episode_reward = 0.0
            episode_transitions = []
            intervention_steps = 0

            print(f"[Actor] Episode {episode + 1}/{MAX_EPISODES}")

            for step in range(MAX_EPISODE_STEPS):
                step_start = time.perf_counter()
                if shutdown_event.is_set():
                    break

                # Check for updated policy params from learner (sent as numpy)
                try:
                    new_params_np = parameters_queue.get_nowait()
                    new_params = {k: torch.from_numpy(v) for k, v in new_params_np.items()}
                    policy.load_state_dict(new_params)
                    print("[Actor] Updated policy parameters from learner")

                except Empty:
                    pass

                # Check for human intervention FIRST (before expensive policy inference)
                teleop_events = teleop.get_teleop_events()
                is_intervention = teleop_events[TeleopEvents.IS_INTERVENTION]
                success = teleop_events[TeleopEvents.SUCCESS]
                terminate = teleop_events[TeleopEvents.TERMINATE_EPISODE]

                if is_intervention:
                    # Human is controlling — skip policy inference entirely for smooth teleop
                    action = teleop.get_action()  # np array [dx, dy, dz, dyaw, gripper]
                    intervention_steps += 1
                else:
                    # Policy controls — run inference
                    policy_obs = obs_to_policy_input(obs, device=actor_device)
                    with torch.no_grad():
                        action_tensor = policy.select_action(policy_obs)
                    action = action_tensor.squeeze(0).cpu().numpy()

                # Step env
                next_obs, _reward, _terminated, _truncated, _info = env.step(action)

                # Reward from human feedback
                reward = 1.0 if success else 0.0
                done = success or terminate
                truncated = (step >= MAX_EPISODE_STEPS - 1)

                # Build observation tensors for replay buffer (always needed for training)
                policy_obs_np = {k: v for k, v in obs_to_policy_input(obs, device="cpu").items()}
                next_policy_obs_np = {k: v for k, v in obs_to_policy_input(next_obs, device="cpu").items()}
                transition = {
                    "state": {k: v.numpy() for k, v in policy_obs_np.items()},
                    "action": action.copy() if isinstance(action, np.ndarray) else action.cpu().numpy(),
                    "reward": float(reward),
                    "next_state": {k: v.numpy() for k, v in next_policy_obs_np.items()},
                    "done": done or truncated,
                    "truncated": truncated,
                    "complementary_info": {
                        "is_intervention": is_intervention,
                    },
                }
                episode_transitions.append(transition)
                episode_reward += reward
                obs = next_obs

                # Maintain FPS
                elapsed = time.perf_counter() - step_start
                time.sleep(max(dt - elapsed, 0.0))

                if done:
                    break

            # Send transitions to learner
            intervention_rate = intervention_steps / max(step + 1, 1)

            print(f"[Actor] Episode {episode + 1} done: "
                  f"reward={episode_reward:.1f}, steps={step + 1}, "
                  f"intervention_rate={intervention_rate:.1%}"
                  )
            try:
                transitions_queue.put(episode_transitions, timeout=5)

            except Full:
                print("[Actor] Transitions queue full, dropping episode")

    except KeyboardInterrupt:
        print("[Actor] Interuppted")

    finally:
        env.close()
        teleop.disconnect()
        print("[Actor] Finished")


# Learner

def drain_transitions(transitions_queue, online_buffer, offline_buffer):
    """Non-blocking: drain all available episodes from queue into buffers."""
    count = 0
    while not transitions_queue.empty():
        try:
            episode_transitions = transitions_queue.get_nowait()
        except Empty:
            break
        for t in episode_transitions:
            t_torch = {
                "state": {k: torch.from_numpy(v) for k, v in t["state"].items()},
                "action": torch.from_numpy(t["action"]),
                "reward": t["reward"],
                "next_state": {k: torch.from_numpy(v) for k, v in t["next_state"].items()},
                "done": t["done"],
                "truncated": t["truncated"],
                "complementary_info": t["complementary_info"],
            }
            online_buffer.add(**t_torch)
            if t["complementary_info"].get("is_intervention", False):
                offline_buffer.add(**t_torch)
            count += 1
    return count


def run_learner(
        transitions_queue: mp.Queue,
        parameters_queue: mp.Queue,
        shutdown_event: mp.Event,
        ):
    """Train SAC policy on transitions streamed from the Actor.
    Trains CONTINUOUSLY on buffer data — does not block waiting for new transitions."""

    policy = SACPolicy(policy_config)
    policy.train()
    policy.to(device=DEVICE)

    optim_params = policy.get_optim_params()
    optimizers = {
        "critic": optim.Adam(optim_params["critic"], lr=LEARNING_RATE),
        "actor": optim.Adam(optim_params["actor"], lr=LEARNING_RATE),
        "temperature": optim.Adam([optim_params["temperature"]], lr=LEARNING_RATE),
    }

    state_keys = [OBS_STATE, f"{OBS_IMAGES}.front", f"{OBS_IMAGES}.side", f"{OBS_IMAGES}.gripper"]
    online_buffer = ReplayBuffer(
        capacity=ONLINE_BUFFER_CAPACITY, device=DEVICE,
        state_keys=state_keys, storage_device="cpu",
    )

    # Load pre-recorded demos into the offline buffer (bootstrapping)
    if DEMO_DATASET_REPO_ID is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        print(f"[Learner] Loading demos from {DEMO_DATASET_REPO_ID}...")
        demo_dataset = LeRobotDataset(repo_id=DEMO_DATASET_REPO_ID, download_videos=False)
        offline_buffer = ReplayBuffer.from_lerobot_dataset(
            lerobot_dataset=demo_dataset, device=DEVICE,
            state_keys=state_keys, storage_device="cpu",
            capacity=OFFLINE_BUFFER_CAPACITY,
        )
        print(f"[Learner] Loaded {len(offline_buffer)} demo transitions into offline buffer")
    else:
        offline_buffer = ReplayBuffer(
            capacity=OFFLINE_BUFFER_CAPACITY, device=DEVICE,
            state_keys=state_keys, storage_device="cpu",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_step = 0

    print("[Learner] Ready, waiting for initial transitions...")

    while not shutdown_event.is_set():
        # 1. Non-blocking: drain any available transitions from actor
        added = drain_transitions(transitions_queue, online_buffer, offline_buffer)
        if added > 0:
            print(
                f"[Learner] +{added} transitions. "
                f"Buffers: online={len(online_buffer)}, offline={len(offline_buffer)}"
            )

        # 2. Wait until buffer has enough data
        if len(online_buffer) < ONLINE_STEPS_BEFORE_LEARNING:
            time.sleep(0.01)  # Avoid CPU spin while waiting for initial data
            continue

        # 3. Sample batch (online + offline mix)
        batch = online_buffer.sample(BATCH_SIZE // 2)
        if len(offline_buffer) >= BATCH_SIZE // 2:
            offline_batch = offline_buffer.sample(BATCH_SIZE // 2)
            batch = concatenate_batch_transitions(batch, offline_batch)

        # 4. Move to device
        actions = batch[ACTION].to(device=DEVICE)
        rewards = batch["reward"].to(device=DEVICE)
        observations = {k: v.to(device=DEVICE) for k, v in batch["state"].items()}
        next_observations = {k: v.to(device=DEVICE) for k, v in batch["next_state"].items()}
        done = batch["done"].to(device=DEVICE)

        # 5. Observation features: only pre-compute when encoder is FROZEN
        #    When freeze=False, pass None — forward() computes internally with gradients
        observation_features = None
        next_observation_features = None
        if (policy.config.vision_encoder_name is not None
                and policy.config.freeze_vision_encoder
                and policy.shared_encoder
                and policy.actor.encoder.has_images):
            with torch.no_grad():
                observation_features = policy.actor.encoder.get_cached_image_features(observations)
                next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations)

        forward_batch = {
            ACTION: actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }

        # --- Critic update ---
        critic_output = policy.forward(forward_batch, model="critic")
        optimizers["critic"].zero_grad()
        critic_output["loss_critic"].backward()
        torch.nn.utils.clip_grad_norm_(policy.critic_ensemble.parameters(), GRAD_CLIP_NORM)
        optimizers["critic"].step()

        # --- Actor + Temperature update (at specified frequency) ---
        if training_step % POLICY_UPDATE_FREQ == 0:
            for _ in range(POLICY_UPDATE_FREQ):
                actor_output = policy.forward(forward_batch, model="actor")
                optimizers["actor"].zero_grad()
                actor_output["loss_actor"].backward()
                torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), GRAD_CLIP_NORM)
                optimizers["actor"].step()

                temp_output = policy.forward(forward_batch, model="temperature")
                optimizers["temperature"].zero_grad()
                temp_output["loss_temperature"].backward()
                torch.nn.utils.clip_grad_norm_([policy.log_alpha], GRAD_CLIP_NORM)
                optimizers["temperature"].step()

        # --- Target network soft update (AFTER all updates) ---
        policy.update_target_networks()

        training_step += 1

        if training_step % 100 == 0:
            print(
                f"[Learner] Step {training_step}: "
                f"critic={critic_output['loss_critic'].item():.4f}"
            )

        # Send updated parameters to actor (as numpy to avoid fd leaks)
        if training_step % SEND_PARAMS_EVERY == 0:
            try:
                state_dict = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
                parameters_queue.put_nowait(state_dict)
            except Full:
                pass

        # Save checkpoint
        if training_step % SAVE_EVERY == 0:
            save_path = OUTPUT_DIR / f"checkpoint_{training_step}"
            policy.save_pretrained(save_path)
            print(f"[Learner] Checkpoint saved at {save_path}")

    # Final save
    policy.save_pretrained(OUTPUT_DIR / "final")
    print("[Learner] Finished")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transitions_queue = mp.Queue(maxsize=10)
    parameters_queue = mp.Queue(maxsize=2)
    shutdown_event = mp.Event()

    def signal_handler(sig, frame):
        print(f"\nSignal {sig} received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    learner_process = mp.Process(
        target=run_learner,
        args=(transitions_queue, parameters_queue, shutdown_event),
        daemon=True,
    )

    actor_process = mp.Process(
        target=run_actor,
        args=(transitions_queue, parameters_queue, shutdown_event),
        daemon=True,
    )

    learner_process.start()
    actor_process.start()

    print("="*50)
    print(" RC1 HIL-SERL training")
    print(" Actor and Learner running...")
    print(" Press Ctrl+C to stop")
    print("="*50)

    try:
        actor_process.join()
        shutdown_event.set()
        learner_process.join(timeout=10)

    except KeyboardInterrupt:
        shutdown_event.set()
        actor_process.join(timeout=5)
        learner_process.join(timeout=10)
    finally:
        if learner_process.is_alive():
            learner_process.terminate()
        if actor_process.is_alive():
            actor_process.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
