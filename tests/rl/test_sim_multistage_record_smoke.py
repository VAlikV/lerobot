"""End-to-end record smoke with stage annotation.

Records a tiny 1-episode dataset on sim_assembling while a fake teleop
flips STAGE_ADVANCE=True at known frames. Asserts the dataset gets
sparse_subtask_* columns + temporal_proportions_{sparse,dense}.json.

Slow: boots MuJoCo (EGL) + writes a dataset to a tmp dir. Skipped unless
a MUJOCO_GL backend is set.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


class _FakeGamepadTeleop:
    """Minimal Teleop stub: never intervenes, but emits STAGE_ADVANCE on a
    schedule. Acts as both the teleop passed into make_processors and the
    record loop."""

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


@pytest.fixture
def tmp_dataset_root(tmp_path):
    root = tmp_path / "smoke_ds"
    yield root
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def test_record_one_ep_with_stage_annotation(tmp_dataset_root):
    try:
        import gymnasium as gym
        import lerobot.envs.sim_assembling  # noqa: F401
    except Exception as e:
        pytest.skip(f"sim_assembling env unavailable: {e}")

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

    env = gym.make(
        "sim_assembling/AssembleBase-v0",
        control_hz=20.0,
        mode="fast",
        render_mode="rgb_array",
        max_episode_steps=6,
        use_gripper=True,
        include_yaw_slot=False,
    )

    teleop = _FakeGamepadTeleop(advance_frames=[2, 4])

    env_cfg = HILSerlRobotEnvConfig(
        name="sim_assembling",
        task="AssembleBase-v0",
        fps=20,
        robot=None,
        teleop=None,  # we pass the fake teleop directly into make_processors
        processor=HILSerlProcessorConfig(
            control_mode="gamepad",
            gripper=GripperConfig(use_gripper=True),
            reset=ResetConfig(terminate_on_success=True, control_time_s=0.3),
            stage_names=["a", "b", "c"],
        ),
    )
    cfg = GymManipulatorConfig(
        env=env_cfg,
        dataset=DatasetConfig(
            repo_id="local/smoke_stage_test",
            task="smoke",
            root=str(tmp_dataset_root),
            num_episodes_to_record=1,
            overwrite=True,
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

    # Validate dataset layout + stage annotations.
    assert tmp_dataset_root.exists(), "dataset dir not created"
    sparse_props = tmp_dataset_root / "meta" / "temporal_proportions_sparse.json"
    dense_props = tmp_dataset_root / "meta" / "temporal_proportions_dense.json"
    assert sparse_props.exists(), "temporal_proportions_sparse.json missing"
    assert dense_props.exists(), "temporal_proportions_dense.json missing"

    props = json.loads(sparse_props.read_text())
    assert set(props.keys()) == {"a", "b", "c"}
    assert abs(sum(props.values()) - 1.0) < 1e-3

    ep_parquets = list((tmp_dataset_root / "meta" / "episodes").rglob("*.parquet"))
    assert ep_parquets, "no episodes parquet found"

    import pandas as pd

    df = pd.read_parquet(ep_parquets[0])
    assert "sparse_subtask_names" in df.columns
    assert "dense_subtask_names" in df.columns
    row_names = list(df.iloc[0]["sparse_subtask_names"])
    # At least stage "a" was seeded + we fired two advances → all 3 stages.
    assert row_names == ["a", "b", "c"]
