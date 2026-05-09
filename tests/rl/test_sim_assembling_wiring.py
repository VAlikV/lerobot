"""Verify make_robot_env + make_processors picks up reward_model for sim_assembling."""

from __future__ import annotations

import os

import pytest


def _skip_if_no_gl():
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture()
def sim_cfg_factory():
    _skip_if_no_gl()
    try:
        import lerobot.envs.sim_assembling  # noqa: F401
    except ImportError:
        pytest.skip("simulator_for_il_rl not installed")

    from lerobot.envs.configs import (
        EnvRewardModelConfig,
        GripperConfig,
        HILSerlProcessorConfig,
        HILSerlRobotEnvConfig,
        ImagePreprocessingConfig,
        ResetConfig,
    )

    def build(reward_model_type: str | None):
        rm = None if reward_model_type is None else EnvRewardModelConfig(
            type=reward_model_type, pretrained_path=None, task="assemble"
        )
        return HILSerlRobotEnvConfig(
            name="sim_assembling",
            task="AssembleBase-v0",
            fps=20,
            realtime=False,
            robot=None,
            teleop=None,
            processor=HILSerlProcessorConfig(
                control_mode="keyboard",
                gripper=GripperConfig(use_gripper=True, gripper_penalty=0.0),
                image_preprocessing=ImagePreprocessingConfig(crop_params_dict=None, resize_size=None),
                reset=ResetConfig(reset_time_s=0.0, control_time_s=1.0, terminate_on_success=True),
                reward_classifier=None,
                reward_model=rm,
                max_gripper_pos=100.0,
            ),
        )

    return build


@pytest.mark.parametrize(
    "reward_type, expected_step",
    [
        (None, None),
        ("manual", None),
        ("height_gripper", "HeightGripperRewardStep"),
        ("cnn", "CNNRewardProcessorStep"),
        ("sarm", "SARMRewardProcessorStep"),
    ],
)
def test_wiring(sim_cfg_factory, reward_type, expected_step):
    from lerobot.processor.reward_model.base import BaseRewardProcessorStep
    from lerobot.rl.gym_manipulator import make_processors, make_robot_env

    cfg = sim_cfg_factory(reward_type)
    try:
        env, teleop = make_robot_env(cfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"env build failed (mujoco/gl): {e}")

    try:
        env_pipe, _ = make_processors(env, teleop, cfg, "cpu")
    finally:
        env.close()

    reward_steps = [s for s in env_pipe.steps if isinstance(s, BaseRewardProcessorStep)]
    if expected_step is None:
        assert reward_steps == []
    else:
        assert len(reward_steps) == 1
        assert type(reward_steps[0]).__name__ == expected_step
