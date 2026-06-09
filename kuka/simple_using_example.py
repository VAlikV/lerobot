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
    ImageCropResizeProcessorStep,
    Numpy2TorchActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
)
from lerobot.processor.converters import create_transition
from lerobot.processor.converters import identity_transition
from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaRobotEnv, KukaIiwaRobotEnvConfig
from lerobot.rl.gym_manipulator import GymManipulatorConfig
from lerobot.utils.robot_utils import precise_sleep


CONFIG_PATH = "kuka/configs/kuka_redpag_record.json"
MODEL_ID = "outputs/test/act/40000"
DATASET_ID = "local/kuka_iiwa_3cams_abs_redpag"
DEVICE = "cuda"

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 1000


def _make_kuka_env(cfg: GymManipulatorConfig) -> KukaIiwaRobotEnv:
    env_cfg = cfg.env
    robot = KukaIiwa(env_cfg.robot)
    robot.connect()

    ik_cfg = env_cfg.processor.inverse_kinematics
    reset_cfg = env_cfg.processor.reset
    use_gripper = env_cfg.processor.gripper.use_gripper if env_cfg.processor.gripper is not None else True
    use_yaw = bool(getattr(ik_cfg, "use_yaw", False)) if ik_cfg else False

    home_tcp = list(reset_cfg.fixed_reset_joint_positions) if reset_cfg else [0.5, 0.0, 0.4, 0.0, 0.0, 0.0]

    env_config = KukaIiwaRobotEnvConfig(
        ee_step_sizes=ik_cfg.end_effector_step_sizes if ik_cfg else {"x": 0.001, "y": 0.001, "z": 0.001},
        ee_bounds_min=ik_cfg.end_effector_bounds["min"]
        if ik_cfg and ik_cfg.end_effector_bounds
        else [-0.5, -0.5, 0.05],
        ee_bounds_max=ik_cfg.end_effector_bounds["max"]
        if ik_cfg and ik_cfg.end_effector_bounds
        else [0.5, 0.5, 0.7],
        fixed_roll=home_tcp[3] if len(home_tcp) > 3 else 0.0,
        fixed_pitch=home_tcp[4] if len(home_tcp) > 4 else 0.0,
        fixed_yaw=home_tcp[5] if len(home_tcp) > 5 else 0.0,
        home_tcp=home_tcp,
        reset_time_s=reset_cfg.reset_time_s if reset_cfg else 5.0,
        reset_fps=getattr(env_cfg.robot, "reset_fps", 30),
        use_gripper=use_gripper,
        use_yaw=use_yaw,
        randomization_xy=reset_cfg.randomization_xy if reset_cfg else 0.0,
        randomization_z=reset_cfg.randomization_z if reset_cfg else 0.0,
    )
    return KukaIiwaRobotEnv(robot, env_config)


def _make_env_processor(cfg: GymManipulatorConfig, device: str):
    steps = [
        Numpy2TorchActionProcessorStep(),
        VanillaObservationProcessorStep(),
    ]

    image_cfg = cfg.env.processor.image_preprocessing
    if image_cfg is not None:
        steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=image_cfg.crop_params_dict,
                resize_size=image_cfg.resize_size,
            )
        )

    steps.append(AddBatchDimensionProcessorStep())
    steps.append(DeviceProcessorStep(device=device))

    return DataProcessorPipeline[EnvTransition, EnvTransition](
        steps=steps,
        to_transition=identity_transition,
        to_output=identity_transition,
    )


def _apply_absolute_action_to_env(env, action: dict[str, float]) -> None:
    """Send an ACT absolute task-space action and keep env bookkeeping coherent."""
    env.robot.send_action(action)

    if hasattr(env, "target_xyz"):
        env.target_xyz = torch.tensor(
            [action["x.pos"], action["y.pos"], action["z.pos"]],
            dtype=torch.float32,
        ).numpy()

    if hasattr(env, "target_yaw") and "yaw.pos" in action:
        fixed_yaw = float(getattr(env.config, "fixed_yaw", 0.0))
        env.target_yaw = float(action["yaw.pos"]) - fixed_yaw if getattr(env, "use_yaw", False) else float(
            action["yaw.pos"]
        )


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")

    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    cfg.device = str(device)

    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)

    policy = ACTPolicy.from_pretrained(MODEL_ID)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=MODEL_ID,
        dataset_stats=dataset_metadata.stats,
    )

    env = _make_kuka_env(cfg)
    env_processor = _make_env_processor(cfg, device=str(device))
    fps = cfg.env.fps or 30
    dt_s = 1.0 / float(fps)

    try:
        for episode_idx in range(MAX_EPISODES):
            obs, info = env.reset()
            env_processor.reset()
            policy.reset()

            for step_idx in range(MAX_STEPS_PER_EPISODE):
                start_t = time.perf_counter()

                transition = create_transition(observation=obs, info=info)
                transition = env_processor(transition)
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
                _apply_absolute_action_to_env(env, robot_action)

                obs = env._get_observation()
                info = {}

                precise_sleep(max(dt_s - (time.perf_counter() - start_t), 0.0))

            print(f"Episode {episode_idx + 1} finished.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
