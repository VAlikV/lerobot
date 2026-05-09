"""Verify make_processors threads StageAnnotatorProcessorStep into the
action pipeline when cfg.processor.stage_names is set.

We build the sim_assembling env with a dummy (mock) teleop so SDL/pygame is
not touched.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")


class _DummyTeleop:
    """Minimal teleop that satisfies AddTeleopActionAsComplimentaryDataStep
    + AddTeleopEventsAsInfoStep protocols."""

    def __init__(self):
        pass

    def get_action(self):
        return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0}

    def get_teleop_events(self):
        from lerobot.teleoperators.utils import TeleopEvents

        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
            TeleopEvents.STAGE_ADVANCE: False,
        }


@pytest.fixture(scope="module")
def sim_env():
    try:
        import gymnasium as gym
        import lerobot.envs.sim_assembling  # noqa: F401
    except Exception as e:
        pytest.skip(f"sim_assembling env unavailable: {e}")
    try:
        env = gym.make(
            "sim_assembling/AssembleBase-v0",
            control_hz=20.0,
            mode="fast",
            render_mode="rgb_array",
            max_episode_steps=5,
            use_gripper=True,
            include_yaw_slot=False,
        )
    except Exception as e:
        pytest.skip(f"Could not construct sim_assembling env: {e}")
    yield env
    env.close()


def _build_cfg(stage_names):
    from lerobot.envs.configs import (
        GripperConfig,
        HILSerlProcessorConfig,
        HILSerlRobotEnvConfig,
        ResetConfig,
    )

    return HILSerlRobotEnvConfig(
        name="sim_assembling",
        task="AssembleBase-v0",
        fps=20,
        robot=None,
        teleop=None,
        processor=HILSerlProcessorConfig(
            control_mode="gamepad",
            gripper=GripperConfig(use_gripper=True),
            reset=ResetConfig(terminate_on_success=True, control_time_s=5.0),
            stage_names=stage_names,
        ),
    )


def test_stage_annotator_inserted_when_stage_names_set(sim_env):
    from lerobot.processor.stage_annotator import StageAnnotatorProcessorStep
    from lerobot.rl.gym_manipulator import make_processors

    cfg = _build_cfg(stage_names=["a", "b", "c"])
    _env_p, action_p = make_processors(sim_env, _DummyTeleop(), cfg, device="cpu")
    steps = list(action_p.steps)
    matches = [s for s in steps if isinstance(s, StageAnnotatorProcessorStep)]
    assert len(matches) == 1
    assert matches[0].stage_names == ["a", "b", "c"]


def test_stage_annotator_absent_when_stage_names_none(sim_env):
    from lerobot.processor.stage_annotator import StageAnnotatorProcessorStep
    from lerobot.rl.gym_manipulator import make_processors

    cfg = _build_cfg(stage_names=None)
    _env_p, action_p = make_processors(sim_env, _DummyTeleop(), cfg, device="cpu")
    assert not any(
        isinstance(s, StageAnnotatorProcessorStep) for s in action_p.steps
    )


def test_stage_index_propagates_through_pipeline(sim_env):
    """A single step() through the pipeline must stamp info[stage_index]=0."""
    from lerobot.processor.core import TransitionKey
    from lerobot.processor.converters import create_transition
    from lerobot.rl.gym_manipulator import make_processors

    cfg = _build_cfg(stage_names=["a", "b"])
    env_p, action_p = make_processors(sim_env, _DummyTeleop(), cfg, device="cpu")
    action_p.reset()

    obs, info = sim_env.reset()
    t = create_transition(observation=obs, info=info or {}, complementary_data={})
    t = env_p(t)
    t[TransitionKey.ACTION.value] = torch.zeros(4)
    t[TransitionKey.ACTION.value][-1] = 1.0  # gripper stay
    t = action_p(t)
    assert t[TransitionKey.INFO.value].get("stage_index") == 0
    assert t[TransitionKey.INFO.value].get("stage_name") == "a"
