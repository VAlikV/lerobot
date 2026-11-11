import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    path = Path(__file__).resolve().parents[3] / "examples/kuka_iiwa/rollout_record.py"
    assert path.exists()
    spec = importlib.util.spec_from_file_location("rollout_record", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_records_fresh_observation_and_sent_action():
    mod = module()
    frames = []
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["j"]},
    }
    robot = SimpleNamespace(
        get_observation=lambda: {"j": 2.0}, send_action=lambda action: {"j": action["j"] - 1}
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(use_relative_actions=False), select_action=lambda obs: torch.tensor([[5.0]])
    )
    dataset = SimpleNamespace(features=features, add_frame=frames.append)
    events = {"stop_recording": False, "exit_early": False, "rerecord_episode": False}
    mod.record_episode(
        robot,
        policy,
        lambda x: x,
        lambda x: x,
        lambda x: x,
        dataset,
        features,
        events,
        "task",
        30,
        1,
        "cpu",
        max_steps=1,
    )
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0]["observation.state"], [2])
    np.testing.assert_array_equal(frames[0]["action"], [4])


def test_stop_before_robot_access():
    mod = module()
    mod.record_episode(
        None, None, None, None, None, None, None, {"stop_recording": True}, "task", 30, 1, "cpu"
    )


def test_rejects_relative_temporal_ensemble():
    mod = module()
    cfg = SimpleNamespace(use_relative_actions=True, temporal_ensemble_coeff=0.01)
    with pytest.raises(ValueError, match="relative"):
        mod.validate_execution(cfg)


def test_relative_queue_records_new_observations_without_reanchoring():
    mod = module()
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["j"]},
    }
    observations = iter([10.0, 20.0])
    frames = []
    anchors = []

    def pre(obs):
        anchors.append(obs["observation.state"].item())
        return obs

    policy = SimpleNamespace(
        config=SimpleNamespace(use_relative_actions=True, n_action_steps=2),
        predict_action_chunk=lambda obs: torch.tensor([[[1.0], [2.0]]]),
    )
    robot = SimpleNamespace(get_observation=lambda: {"j": next(observations)}, send_action=lambda x: x)
    mod.record_episode(
        robot,
        policy,
        pre,
        lambda x: x + anchors[-1],
        lambda x: x,
        SimpleNamespace(features=features, add_frame=frames.append),
        features,
        {"stop_recording": False},
        "task",
        30,
        2,
        "cpu",
        max_steps=2,
    )
    assert anchors == [10.0]
    assert [f["observation.state"].item() for f in frames] == [10.0, 20.0]
    assert [f["action"].item() for f in frames] == [11.0, 12.0]


def test_cleanup_disconnects_partial_connection_and_finalizes_on_error():
    mod = module()
    calls = []

    def disconnect():
        calls.append("disconnect")
        raise RuntimeError("disconnect failed")

    robot = SimpleNamespace(is_connected=False, disconnect=disconnect)
    listener = SimpleNamespace(stop=lambda: calls.append("listener"))
    dataset = SimpleNamespace(
        has_pending_frames=lambda: True,
        clear_episode_buffer=lambda: calls.append("discard"),
        finalize=lambda: calls.append("finalize"),
    )
    with pytest.raises(RuntimeError, match="disconnect failed"):
        mod.cleanup(robot, listener, dataset)
    assert calls == ["disconnect", "listener", "discard", "finalize"]
