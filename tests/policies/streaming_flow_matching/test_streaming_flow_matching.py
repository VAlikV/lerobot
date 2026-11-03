from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.streaming_flow_matching.configuration_streaming_flow_matching import (
    StreamingFlowMatchingConfig,
)
from lerobot.policies.streaming_flow_matching.flow import interpolate_trajectory, sample_conditional_flow
from lerobot.policies.streaming_flow_matching.modeling_streaming_flow_matching import (
    StreamingFlowMatchingPolicy,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    return StreamingFlowMatchingConfig(
        device="cpu",
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        down_dims=(8, 16),
        n_groups=4,
        time_embed_dim=8,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (3,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        state_feature_names=["gripper.pos", "unused", "joint_1.pos"],
        action_feature_names=["joint_1.pos", "gripper.pos"],
        **kwargs,
    )


def stats():
    return {
        "observation.state": {
            "min": torch.tensor([-1.0, 0.0, 10.0]),
            "max": torch.tensor([1.0, 100.0, 20.0]),
        },
        "action": {"min": torch.tensor([-100.0, -100.0]), "max": torch.tensor([100.0, 100.0])},
    }


def training_batch():
    state = torch.tensor(
        [[[-1.0, 2.0, 10.0], [-1.0, 3.0, 12.0], [1.0, 4.0, 14.0], [1.0, 5.0, 16.0], [1.0, 6.0, 18.0]]]
    )
    return {
        "observation.state": state,
        "observation.state_is_pad": torch.zeros(1, 5, dtype=torch.bool),
        "action": torch.full((1, 2), 99.0),
    }


def observation(joint=12.0):
    return {"observation.state": torch.tensor([[-1.0, 3.0, joint]])}


def test_interpolation_and_normalized_derivative():
    trajectory = torch.tensor([[[0.0, 5.0], [2.0, 5.0], [4.0, 5.0], [6.0, 5.0]]])
    a, v = interpolate_trajectory(trajectory, torch.tensor([0.5]))
    torch.testing.assert_close(a, torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(v, torch.tensor([[6.0, 0.0]]))
    a, v = interpolate_trajectory(trajectory, torch.ones(1))
    torch.testing.assert_close(a, trajectory[:, -1])
    torch.testing.assert_close(v, torch.tensor([[6.0, 0.0]]))


def test_deterministic_and_stochastic_flow_formulas():
    trajectory = torch.tensor([[[0.0, 1.0], [2.0, 1.0]]])
    t, eps, z = torch.tensor([0.5]), torch.tensor([[1.0, -1.0]]), torch.tensor([[2.0, 3.0]])
    x, target = sample_conditional_flow(
        trajectory, t, variant="deterministic", sigma=0.2, gain=2.0, noise=eps
    )
    error = 0.2 * torch.exp(-2 * t[:, None]) * eps
    torch.testing.assert_close(x[:, 0], torch.tensor([[1.0, 1.0]]) + error)
    torch.testing.assert_close(target[:, 0], torch.tensor([[2.0, 0.0]]) - 2 * error)
    x, target = sample_conditional_flow(
        trajectory, t, variant="stochastic", sigma=0.1, sigma1=0.2, noise=eps, latent=z
    )
    r = (0.2**2 - 0.1**2) ** 0.5
    torch.testing.assert_close(x[:, 0], torch.tensor([[1.0, 1.0]]) + 0.1 * eps + r * 0.5 * z)
    torch.testing.assert_close(x[:, 1], 0.6 * z + 0.5 * torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(target[:, 0], torch.tensor([[2.0, 0.0]]) + r * z)
    torch.testing.assert_close(target[:, 1], torch.tensor([[2.0, 1.0]]) - 0.8 * z)


def test_processors_use_measured_states_without_future_leakage():
    cfg = config()
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats())
    batch = training_batch()
    result = pre(batch)
    assert result["observation.state"].shape == (1, 2, 3)
    expected = torch.tensor([[[-0.6, -1.0], [-0.2, 1.0], [0.2, 1.0], [0.6, 1.0]]])
    torch.testing.assert_close(result["action"], expected)
    torch.testing.assert_close(pre(observation())["sfp_anchor"], expected[:, 0])
    torch.testing.assert_close(post(expected[:, 0]), torch.tensor([[12.0, -1.0]]))
    changed = {**batch, "action": torch.full((1, 2), -999.0)}
    torch.testing.assert_close(pre(changed)["action"], result["action"])
    changed_state = batch["observation.state"].clone()
    changed_state[:, 2:] += 100
    changed = pre({**batch, "observation.state": changed_state})
    torch.testing.assert_close(changed["observation.state"], result["observation.state"])
    assert not torch.equal(changed["action"], result["action"])
    assert batch["observation.state"].shape == (1, 5, 3)


def test_padding_masks_only_valid_intervals():
    pre, _ = make_pre_post_processors(config(), dataset_stats=stats())
    batch = training_batch()
    batch["observation.state_is_pad"][:, -2:] = True
    result = pre(batch)
    assert result["sfp_valid_intervals"].tolist() == [[True, False, False]]
    policy = StreamingFlowMatchingPolicy(config())
    loss, _ = policy(result)
    assert torch.isfinite(loss)
    batch["observation.state_is_pad"][:, 2:] = True
    with pytest.raises(ValueError, match="valid"):
        policy(pre(batch))


class ConstantField(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, time, conditioning):
        self.calls.append((time.clone(), conditioning.clone(), x.clone()))
        return torch.ones_like(x)


@pytest.mark.parametrize("variant", ["deterministic", "stochastic"])
def test_streaming_chunk_parity_and_reset(variant):
    cfg = config(variant=variant, integration_steps_per_action=3)
    pre, _ = make_pre_post_processors(cfg, dataset_stats=stats())
    policy = StreamingFlowMatchingPolicy(cfg).eval()
    field = ConstantField()
    policy.velocity_net = field
    batch = pre(observation())
    torch.manual_seed(4)
    expected = policy.predict_action_chunk(batch)
    assert policy._flow_state is None
    torch.manual_seed(4)
    first = policy.select_action(batch)
    second = policy.select_action(pre(observation(18.0)))
    torch.testing.assert_close(torch.stack([first, second], 1), expected)
    torch.testing.assert_close(first, batch["sfp_anchor"] + 1 / 3)
    # Fresh measured anchor after n_action_steps, with a new history/conditioning.
    third = policy.select_action(pre(observation(18.0)))
    torch.testing.assert_close(third, pre(observation(18.0))["sfp_anchor"] + 1 / 3)
    assert len(field.calls) == 15
    torch.testing.assert_close(field.calls[6][1], field.calls[9][1])
    policy.reset()
    assert policy._flow_state is None
    assert not policy._observation_history


def test_metadata_validation_and_permuted_names():
    cfg = config()
    features = {
        "observation.state": {"names": ["joint_1.pos", "unused", "gripper.pos"]},
        "action": {"names": ["joint_1.pos", "gripper.pos"]},
    }
    cfg.set_dataset_feature_metadata(features)
    assert cfg.state_action_indices == [0, 2]
    meta = SimpleNamespace(total_frames=0, total_episodes=0)
    with pytest.raises(ValueError, match="empty"):
        cfg.validate_dataset_metadata(meta)
    with pytest.raises(ValueError, match="missing|Missing"):
        cfg.set_dataset_feature_metadata({**features, "observation.state": {"names": ["wrong"]}})


@pytest.mark.parametrize("variant", ["deterministic", "stochastic"])
def test_training_and_checkpoint_roundtrip(tmp_path, variant):
    cfg = config(variant=variant)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats())
    policy = StreamingFlowMatchingPolicy(cfg)
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    for _ in range(3):
        loss, metrics = policy(pre(training_batch()))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        optimizer.step()
        optimizer.zero_grad()
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded = StreamingFlowMatchingPolicy.from_pretrained(tmp_path)
    new_pre, new_post = make_pre_post_processors(loaded.config, pretrained_path=tmp_path)
    torch.testing.assert_close(new_pre(training_batch())["action"], pre(training_batch())["action"])
    torch.testing.assert_close(new_post(torch.zeros(1, 2)), post(torch.zeros(1, 2)))
    loaded.train()
    loss, _ = loaded(new_pre(training_batch()))
    loss.backward()
    assert get_policy_class("streaming_flow_matching") is StreamingFlowMatchingPolicy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 1},
        {"n_action_steps": 4},
        {"n_obs_steps": 0},
        {"integration_steps_per_action": 0},
        {"variant": "wrong"},
        {"sigma": -1.0},
        {"variant": "stochastic", "sigma": 0.2, "sigma1": 0.1},
    ],
)
def test_invalid_config(kwargs):
    values = {"device": "cpu", "horizon": 4, "n_action_steps": 2}
    values.update(kwargs)
    with pytest.raises(ValueError):
        StreamingFlowMatchingConfig(**values)


def test_every_tick_feedback_and_active_preview_are_isolated():
    cfg = config()
    cfg.n_action_steps = 1
    pre, _ = make_pre_post_processors(cfg, dataset_stats=stats())
    policy = StreamingFlowMatchingPolicy(cfg).eval()
    field = ConstantField()
    policy.velocity_net = field
    policy.select_action(pre(observation(12.0)))
    before = policy._flow_state.sample.clone()
    history = policy._observation_history["observation.state"][-1].clone()
    policy.predict_action_chunk(pre(observation(18.0)))
    torch.testing.assert_close(policy._flow_state.sample, before)
    torch.testing.assert_close(policy._observation_history["observation.state"][-1], history)
    action = policy.select_action(pre(observation(18.0)))
    torch.testing.assert_close(action, pre(observation(18.0))["sfp_anchor"] + 1 / 3)
    assert not torch.equal(field.calls[0][1], field.calls[-1][1])


def test_latent_is_carried_and_eight_tick_conditioning():
    cfg = config(variant="stochastic")
    cfg.horizon, cfg.n_action_steps = 16, 8
    pre, _ = make_pre_post_processors(cfg, dataset_stats=stats())
    policy = StreamingFlowMatchingPolicy(cfg).eval()
    field = ConstantField()
    policy.velocity_net = field
    for step in range(8):
        policy.select_action(pre(observation(12.0 + step)))
    assert len(field.calls) == 8
    for call in field.calls:
        torch.testing.assert_close(call[1], field.calls[0][1])
    for previous, current in zip(field.calls[:-1], field.calls[1:], strict=True):
        torch.testing.assert_close(current[2][:, 1], previous[2][:, 1] + 1 / 15)
    policy.select_action(pre(observation(14.0)))
    assert field.calls[-1][0].item() == 0
    assert not torch.equal(field.calls[-1][1], field.calls[0][1])


def test_finetune_overrides_keep_measured_normalization(tmp_path):
    cfg = config()
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats())
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = make_pre_post_processors(
        cfg,
        pretrained_path=tmp_path,
        preprocessor_overrides={"normalizer_processor": {"stats": stats()}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats()}},
    )
    torch.testing.assert_close(loaded_pre(observation())["sfp_anchor"], pre(observation())["sfp_anchor"])
    torch.testing.assert_close(loaded_post(torch.zeros(1, 2)), post(torch.zeros(1, 2)))


def test_three_camera_training_and_history():
    cfg = config()
    cfg.resize_shape = (32, 32)
    cfg.vision_feature_dim = 4
    for name in ("front", "side", "gripper"):
        cfg.input_features[f"observation.images.{name}"] = PolicyFeature(FeatureType.VISUAL, (3, 32, 32))
    all_stats = stats()
    batch = training_batch()
    frame = observation()
    for key in cfg.image_features:
        all_stats[key] = {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}
        batch[key] = torch.rand(1, 2, 3, 32, 32)
        frame[key] = torch.rand(1, 3, 32, 32)
    pre, _ = make_pre_post_processors(cfg, dataset_stats=all_stats)
    policy = StreamingFlowMatchingPolicy(cfg)
    loss, _ = policy(pre(batch))
    loss.backward()
    assert all(encoder.conv1.weight.grad is not None for encoder in policy.observation_encoder.cameras)
    policy.eval()
    assert policy.select_action(pre(frame)).shape == (1, 2)


@pytest.mark.parametrize("variant", ["deterministic", "stochastic"])
def test_real_lerobot_dataset_and_standard_trainer_resume(tmp_path, monkeypatch, variant):
    import sys

    import numpy as np

    from lerobot.configs.default import DatasetConfig, EMAConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.lerobot_train import train

    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/sfp_synthetic",
        fps=30,
        root=root,
        use_videos=False,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["gripper.pos", "unused", "joint_1.pos"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["joint_1.pos", "gripper.pos"]},
        },
    )
    for length in (1, 3, 6, 2):
        for i in range(length):
            dataset.add_frame(
                {
                    "observation.state": np.array([-1.0, i, 10.0 + i], dtype=np.float32),
                    "action": np.array([999.0, 999.0], dtype=np.float32),
                    "task": "test SFP",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    cfg = config(variant=variant)
    cfg.push_to_hub = False
    pipeline = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/sfp_synthetic", root=str(root), eval_split=0.25),
        policy=cfg,
        output_dir=tmp_path / "training",
        batch_size=1,
        eval_steps=1,
        steps=2,
        num_workers=0,
        env_eval_freq=0,
        log_freq=1,
        save_freq=1,
        ema=EMAConfig(enable=True),
    )
    # Actual loader, timestamps, episode padding and named metadata hook.
    loaded_dataset = make_dataset(pipeline)
    pre, _ = make_pre_post_processors(cfg, dataset_stats=loaded_dataset.meta.stats)
    sample = loaded_dataset[1]
    batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in sample.items()
    }
    processed = pre(batch)
    assert processed["sfp_valid_intervals"].tolist() == [[True, True, False]]
    torch.testing.assert_close(processed["observation.state"][:, 0], processed["observation.state"][:, 1])
    assert processed["action"][0, :, 1].isfinite().all()  # constant gripper
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])
    train(pipeline)
    checkpoint = tmp_path / "training/checkpoints/last/pretrained_model"
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint.parent / "pretrained_model_ema/model.safetensors").exists()
    # Resume through the same CLI parser used in production, restoring optimizer and EMA.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            f"--config_path={checkpoint / 'train_config.json'}",
            "--resume=true",
            "--steps=3",
            "--max_eval_samples=1",
        ],
    )
    train()
    assert (tmp_path / "training/checkpoints/000003/pretrained_model/model.safetensors").exists()


def test_mock_kuka_action_uses_absolute_degrees_and_safety():
    from examples.kuka_iiwa.rollout_streaming_flow_matching import make_safety_pipeline, predict_robot_action

    names = [f"joint_{i}.pos" for i in range(1, 8)] + ["gripper.pos"]

    class MockPolicy:
        def select_action(self, batch):
            return torch.tensor([[180.0, 130.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])

    features = {"action": {"names": names, "shape": (8,), "dtype": "float32"}}
    result = predict_robot_action(
        MockPolicy(), lambda x: x, lambda x: x, {}, features, make_safety_pipeline()
    )
    assert list(result) == names
    assert result["joint_1.pos"] == 168.0
    assert result["joint_2.pos"] == 118.0
    assert result["gripper.pos"] == -1.0


def test_hardware_entrypoint_with_mock_robot(tmp_path, monkeypatch):
    import json
    import sys

    import examples.kuka_iiwa.record_config as record_config
    import examples.kuka_iiwa.rollout_streaming_flow_matching as rollout
    import lerobot.robots.kuka_iiwa as kuka_module

    names = [f"joint_{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
    features = {
        key: {"names": names, "shape": (8,), "dtype": "float32"} for key in ("observation.state", "action")
    }
    cfg = StreamingFlowMatchingConfig(
        device="cpu",
        state_feature_names=names,
        action_feature_names=names,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (8,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
        down_dims=(8, 16),
        n_groups=4,
        time_embed_dim=8,
    )
    policy = StreamingFlowMatchingPolicy(cfg)
    all_stats = {"observation.state": {"min": -torch.ones(8), "max": torch.ones(8)}}
    pre, post = make_pre_post_processors(cfg, dataset_stats=all_stats)
    metadata = SimpleNamespace(features=features, camera_keys=[], fps=30)
    sent, lifecycle = [], []

    class FakeRobot:
        action_features = dict.fromkeys(names, float)
        observation_features = dict.fromkeys(names, float)
        is_connected = False

        def __init__(self, config):
            pass

        def connect(self):
            self.is_connected = True
            lifecycle.append("connect")

        def get_observation(self):
            return dict.fromkeys(names, 0.0)

        def send_action(self, command):
            sent.append(command)

        def disconnect(self):
            self.is_connected = False
            lifecycle.append("disconnect")

    raw = json.loads(rollout.CONFIG_PATH.read_text())
    raw["robot"]["cameras"] = {}
    config_path = tmp_path / "record_config.json"
    config_path.write_text(json.dumps(raw))
    monkeypatch.setitem(sys.modules, "record_config", record_config)
    monkeypatch.setattr(kuka_module, "KukaIiwa", FakeRobot)
    monkeypatch.setattr(rollout, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(StreamingFlowMatchingPolicy, "from_pretrained", lambda *args, **kwargs: policy)
    monkeypatch.setattr(rollout, "make_pre_post_processors", lambda *args, **kwargs: (pre, post))
    monkeypatch.setattr(rollout, "precise_sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rollout", "--config", str(config_path), "--device=cpu", "--episodes=1", "--max-steps=1"],
    )
    rollout.main()
    assert lifecycle == ["connect", "disconnect"]
    assert len(sent) == 1 and list(sent[0]) == names


def test_stochastic_preview_does_not_change_next_sample():
    cfg = config(variant="stochastic")
    pre, _ = make_pre_post_processors(cfg, dataset_stats=stats())
    batch = pre(observation())
    policy = StreamingFlowMatchingPolicy(cfg).eval()
    torch.manual_seed(7)
    expected = policy.select_action(batch)
    expected_rng = torch.random.get_rng_state()
    policy.reset()
    torch.manual_seed(7)
    policy.predict_action_chunk(batch)
    actual = policy.select_action(batch)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this process")
def test_cuda_default_three_camera_training_and_tick_budget():
    import time

    names = [f"joint_{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
    state_names = names + [f"aux_{i}" for i in range(9)]
    cfg = StreamingFlowMatchingConfig(
        device="cuda",
        state_feature_names=state_names,
        action_feature_names=names,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (17,)),
            **{
                f"observation.images.{name}": PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
                for name in ("front", "side", "gripper")
            },
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    )
    all_stats = {"observation.state": {"min": -torch.ones(17), "max": torch.ones(17)}}
    for key in cfg.image_features:
        all_stats[key] = {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}
    pre, post = make_pre_post_processors(cfg, dataset_stats=all_stats)
    torch.cuda.reset_peak_memory_stats()
    policy = StreamingFlowMatchingPolicy(cfg).cuda()
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    batch = {"observation.state": torch.rand(32, 17, 17), "action": torch.zeros(32, 8)}
    for key in cfg.image_features:
        batch[key] = torch.rand(32, 2, 3, 224, 224)
    loss, _ = policy(pre(batch))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    del batch, loss, optimizer
    policy.eval()
    frame = {"observation.state": torch.zeros(1, 17)}
    for key in cfg.image_features:
        frame[key] = torch.rand(1, 3, 224, 224)
    timings = {}
    for horizon in (8, 1):
        cfg.n_action_steps = horizon
        policy.reset()
        for _ in range(3):
            post(policy.select_action(pre(frame)))
        elapsed = []
        for _ in range(16):
            start = time.perf_counter()
            command = post(policy.select_action(pre(frame)))
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
            assert command.shape == (1, 8) and torch.isfinite(command).all()
        timings[horizon] = {"mean_ms": sum(elapsed) / len(elapsed), "max_ms": max(elapsed)}
    print(
        f"\nSFP CUDA smoke: gpu={torch.cuda.get_device_name()}, batch=32, peak_allocated_gib={peak_gb:.2f}, ticks={timings}"
    )
    del policy
    torch.cuda.empty_cache()
