"""Record 1 ep fresh, then resume & record 1 more ep — verify both land.

Exercises --dataset.resume=true on a tiny sim_assembling dataset. Also
checks stage-annotation offset math: session-2 annotations must land in
shard episode_index = (prior_num_episodes + session_local_idx).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


class _FakeGamepadTeleop:
    def __init__(self, advance_frames):
        self._advance_frames = set(advance_frames)
        self._tick = 0
        self.is_connected = True

    @property
    def action_features(self):
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_action(self):
        return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0}

    def get_teleop_events(self):
        from lerobot.teleoperators.utils import TeleopEvents

        advance = self._tick in self._advance_frames
        self._tick += 1
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
            TeleopEvents.STAGE_ADVANCE: advance,
        }


def _make_env():
    import gymnasium as gym

    import lerobot.envs.sim_assembling  # noqa: F401

    return gym.make(
        "sim_assembling/AssembleBase-v0",
        control_hz=20.0,
        mode="fast",
        render_mode="rgb_array",
        max_episode_steps=6,
        use_gripper=True,
        include_yaw_slot=False,
    )


def _run_session(root: Path, resume: bool, advance_frames, stage_names=("a", "b", "c")):
    from lerobot.envs.configs import (
        GripperConfig,
        HILSerlProcessorConfig,
        HILSerlRobotEnvConfig,
        ResetConfig,
    )
    from lerobot.rl.gym_manipulator import (
        DatasetConfig,
        GymManipulatorConfig,
        control_loop,
        make_processors,
    )

    env = _make_env()
    teleop = _FakeGamepadTeleop(advance_frames=advance_frames)
    env_cfg = HILSerlRobotEnvConfig(
        name="sim_assembling",
        task="AssembleBase-v0",
        fps=20,
        robot=None,
        teleop=None,
        processor=HILSerlProcessorConfig(
            control_mode="gamepad",
            gripper=GripperConfig(use_gripper=True),
            reset=ResetConfig(terminate_on_success=True, control_time_s=0.3),
            stage_names=list(stage_names),
        ),
    )
    cfg = GymManipulatorConfig(
        env=env_cfg,
        dataset=DatasetConfig(
            repo_id="local/smoke_resume_test",
            task="smoke",
            root=str(root),
            num_episodes_to_record=1,
            resume=resume,
        ),
        mode="record",
        device="cpu",
    )
    env_p, action_p = make_processors(env, teleop, env_cfg, device="cpu")
    try:
        control_loop(env, env_p, action_p, teleop, cfg)
    finally:
        try:
            env.close()
        except Exception:
            pass


@pytest.fixture
def ds_root(tmp_path):
    root = tmp_path / "resume_ds"
    yield root
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def test_fresh_then_resume_appends_episode(ds_root):
    try:
        _make_env().close()
    except Exception as e:
        pytest.skip(f"sim_assembling unavailable: {e}")

    import pandas as pd

    def _concat_all_shards():
        shards = sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))
        assert shards, "no episodes parquet"
        return (
            pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
            .sort_values("episode_index")
            .reset_index(drop=True)
        )

    # Session 1: fresh. 1 episode, stage advances at frames {2, 4}.
    _run_session(ds_root, resume=False, advance_frames=[2, 4])
    df1 = _concat_all_shards()
    assert len(df1) == 1
    assert df1.iloc[0]["episode_index"] == 0
    assert list(df1.iloc[0]["sparse_subtask_names"]) == ["a", "b", "c"]

    # Session 2: resume. 1 more episode, different advance pattern (only 1 advance → 2 stages).
    _run_session(ds_root, resume=True, advance_frames=[2])
    df2 = _concat_all_shards()
    assert len(df2) == 2, "session-2 did not append a new episode"
    # Session-1 episode is preserved.
    assert list(df2.iloc[0]["sparse_subtask_names"]) == ["a", "b", "c"]
    # Session-2 episode gets its session-local annotation written to shard idx 1.
    assert df2.iloc[1]["episode_index"] == 1
    assert list(df2.iloc[1]["sparse_subtask_names"]) == ["a", "b"]

    # temporal_proportions_sparse.json reflects ONLY the most recent session.
    props = json.loads((ds_root / "meta" / "temporal_proportions_sparse.json").read_text())
    assert set(props.keys()) == {"a", "b", "c"}
    # Session 2 only visited a, b → proportion of c is 0.
    assert props["c"] == 0.0
    assert props["a"] > 0.0
    assert props["b"] > 0.0


def test_resume_and_overwrite_mutually_exclusive(ds_root, tmp_path):
    from lerobot.envs.configs import (
        GripperConfig,
        HILSerlProcessorConfig,
        HILSerlRobotEnvConfig,
        ResetConfig,
    )
    from lerobot.rl.gym_manipulator import DatasetConfig, GymManipulatorConfig, control_loop

    try:
        env = _make_env()
    except Exception as e:
        pytest.skip(f"sim_assembling unavailable: {e}")
    teleop = _FakeGamepadTeleop(advance_frames=[])
    env_cfg = HILSerlRobotEnvConfig(
        name="sim_assembling",
        task="AssembleBase-v0",
        fps=20,
        robot=None,
        teleop=None,
        processor=HILSerlProcessorConfig(
            control_mode="gamepad",
            gripper=GripperConfig(use_gripper=True),
            reset=ResetConfig(terminate_on_success=True, control_time_s=0.3),
        ),
    )
    cfg = GymManipulatorConfig(
        env=env_cfg,
        dataset=DatasetConfig(
            repo_id="local/smoke_resume_test_x",
            task="smoke",
            root=str(ds_root),
            num_episodes_to_record=1,
            resume=True,
            overwrite=True,
        ),
        mode="record",
        device="cpu",
    )

    from lerobot.rl.gym_manipulator import make_processors

    env_p, action_p = make_processors(env, teleop, env_cfg, device="cpu")
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            control_loop(env, env_p, action_p, teleop, cfg)
    finally:
        env.close()
