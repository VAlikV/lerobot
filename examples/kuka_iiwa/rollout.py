import json
import time

import draccus
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    RobotAction,
    RobotProcessorPipeline,
    TransitionKey,
)
from lerobot.processor.converters import (
    create_transition,
    identity_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.kuka_iiwa import KukaIiwa
from lerobot.robots.kuka_iiwa.robot_kinematic_processor import KukaJointBoundsAndSafety
from lerobot.common.control_utils import follower_smooth_move_to
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from record_config import RecordingConfig

CONFIG_PATH = "lerobot/examples/kuka_iiwa/configs/record_config.json"

MODEL_ID = "outputs/device_assemble"
DATASET_ID = "local/kuka_test_1"
DEVICE = "cuda"

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 10000

N_ACTION_STEPS = 30

RESET_TO_POSE = True


def load_config(path: str) -> RecordingConfig:
    with open(path, "r") as f:
        raw_cfg = json.load(f)
    return draccus.decode(RecordingConfig, raw_cfg)


def _make_runtime_processor(device: str) -> DataProcessorPipeline[EnvTransition, EnvTransition]:
    return DataProcessorPipeline[EnvTransition, EnvTransition](
        steps=[
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ],
        to_transition=identity_transition,
        to_output=identity_transition,
    )


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")

    cfg = load_config(CONFIG_PATH)
    fps = cfg.fps

    reset_pose = cfg.dataset.reset_pose

    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)

    policy = ACTPolicy.from_pretrained(MODEL_ID)
    policy.config.n_action_steps = N_ACTION_STEPS
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=MODEL_ID,
        dataset_stats=dataset_metadata.stats,
    )

    runtime_processor = _make_runtime_processor(device=str(device))

    safety_pipeline = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            KukaJointBoundsAndSafety(joint_offset_deg=cfg.pipeline.joint_offset_deg),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    follower = KukaIiwa(cfg.robot)
    follower.connect()

    dt_s = 1.0 / float(fps)
    print(f"Rolling out {MODEL_ID} on {DATASET_ID} action space, fps={fps}")

    try:
        for episode_idx in range(MAX_EPISODES):
            if RESET_TO_POSE:
                current = follower.get_observation()
                target = dict(zip(follower.action_features.keys(), reset_pose))
                follower_smooth_move_to(follower, current, target, duration_s=3.0)

            policy.reset()
            print(f"\n--- Episode {episode_idx + 1}/{MAX_EPISODES} ---")

            for step_idx in range(MAX_STEPS_PER_EPISODE):
                start_t = time.perf_counter()

                obs = follower.get_observation()
                observation_frame = build_dataset_frame(dataset_metadata.features, obs, prefix=OBS_STR)

                transition = create_transition(observation=observation_frame)
                transition = runtime_processor(transition)
                observation = {
                    key: value
                    for key, value in transition[TransitionKey.OBSERVATION].items()
                    if key in policy.config.input_features
                }
                observation = preprocess(observation)

                with torch.no_grad():
                    action = policy.select_action(observation)
                action = postprocess(action)

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