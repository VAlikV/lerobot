"""End-to-end: gym_manipulator.make_processors picks up reward_model cfg.

Verifies the dispatcher correctly wires a reward step into the env processor
pipeline for each of height_gripper / cnn / sarm / manual, without crashing.

No model checkpoints are loaded (pretrained_path left None); the test only
checks wiring.
"""

from __future__ import annotations

import os

import pytest


def _skip_if_no_gl():
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture(scope="module")
def minimal_env_cfg():
    _skip_if_no_gl()
    try:
        import gym_hil  # noqa: F401
    except ImportError:
        pytest.skip("gym_hil not installed")

    from lerobot.envs.configs import (
        EnvRewardModelConfig,
        GripperConfig,
        HILSerlProcessorConfig,
        HILSerlRobotEnvConfig,
        ImagePreprocessingConfig,
        ResetConfig,
    )

    def build(reward_model: EnvRewardModelConfig | None):
        return HILSerlRobotEnvConfig(
            name="gym_hil",
            task="PandaPickCubeBase-v0",
            fps=10,
            robot=None,
            teleop=None,
            processor=HILSerlProcessorConfig(
                control_mode="keyboard",
                gripper=GripperConfig(use_gripper=True, gripper_penalty=0.0),
                image_preprocessing=ImagePreprocessingConfig(crop_params_dict=None, resize_size=None),
                reset=ResetConfig(reset_time_s=0.0, control_time_s=0.5, terminate_on_success=True),
                reward_classifier=None,
                reward_model=reward_model,
                max_gripper_pos=100.0,
            ),
        )

    return build


def _make_env(cfg):
    """Build the gym_hil env directly. ``make_robot_env`` hardcodes
    ``render_mode="human"`` which needs X, so bypass it for tests."""
    import gym_hil  # noqa: F401
    import gymnasium as gym

    return gym.make(
        f"gym_hil/{cfg.task}",
        image_obs=True,
        render_mode="rgb_array",
    )


def _extract_reward_steps(pipeline):
    from lerobot.processor.reward_model.base import BaseRewardProcessorStep

    return [s for s in pipeline.steps if isinstance(s, BaseRewardProcessorStep)]


@pytest.mark.parametrize(
    "reward_cfg, expected_class_name",
    [
        (None, None),
        ({"type": "manual"}, None),
        ({"type": "height_gripper", "z_index": 2, "gripper_index": 7}, "HeightGripperRewardStep"),
        ({"type": "cnn", "pretrained_path": None}, "CNNRewardProcessorStep"),
        ({"type": "sarm", "pretrained_path": None, "task": "pick"}, "SARMRewardProcessorStep"),
    ],
)
def test_make_processors_wires_reward_model(minimal_env_cfg, reward_cfg, expected_class_name):
    from lerobot.envs.configs import EnvRewardModelConfig

    rm = None if reward_cfg is None else EnvRewardModelConfig(**reward_cfg)
    cfg = minimal_env_cfg(rm)
    try:
        env = _make_env(cfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Env creation failed (mujoco/gl): {e}")

    from lerobot.rl.gym_manipulator import make_processors

    env_pipe, _act_pipe = make_processors(env, None, cfg, "cpu")
    reward_steps = _extract_reward_steps(env_pipe)
    env.close()
    if expected_class_name is None:
        assert reward_steps == []
    else:
        assert len(reward_steps) == 1
        assert type(reward_steps[0]).__name__ == expected_class_name
