from copy import deepcopy

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors


@pytest.fixture
def config(monkeypatch):
    tiny = Dinov2Config(
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=48,
        image_size=28,
        patch_size=14,
    )
    monkeypatch.setattr(Dinov2Model, "from_pretrained", lambda *a, **kw: Dinov2Model(tiny))
    return ACTConfig(
        device="cpu",
        vision_backbone="dinov2",
        chunk_size=3,
        n_action_steps=1,
        dim_model=32,
        dim_feedforward=64,
        n_heads=4,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(17,)),
            **{
                f"observation.images.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28))
                for cam in ("front", "gripper")
            },
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    )


def batch(config):
    return {
        **{k: torch.rand(2, *v.shape) for k, v in config.input_features.items()},
        "action": torch.rand(2, 3, 8),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }


def stats(config):
    return {
        k: {"mean": torch.zeros(v.shape[0]), "std": torch.ones(v.shape[0])}
        for k, v in (config.input_features | config.output_features).items()
    }


def test_frozen_backbone_and_trainable_head(config):
    policy = ACTPolicy(config).train()
    backbone = policy.model.backbone
    assert not backbone.model.training
    assert backbone(torch.rand(2, 3, 28, 28))["feature_map"].shape == (2, 24, 2, 2)
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    projection = policy.model.encoder_img_feat_input_proj.weight.detach().clone()
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    loss, _ = policy(batch(config))
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    assert policy.model.encoder_img_feat_input_proj.weight.grad.abs().sum() > 0
    assert policy.model.action_head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(torch.equal(before[k], v) for k, v in backbone.state_dict().items())
    assert not torch.equal(projection, policy.model.encoder_img_feat_input_proj.weight)
    assert policy.predict_action_chunk(batch(config)).shape == (2, 3, 8)


def test_normalization_and_offline_roundtrip(config, tmp_path, monkeypatch):
    policy = ACTPolicy(config).eval()
    original = stats(config)
    saved = deepcopy(original)
    pre, post = make_act_pre_post_processors(config, original)
    data = batch(config)
    processed = pre(deepcopy(data))
    key = "observation.images.front"
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    torch.testing.assert_close(processed[key], (data[key] - mean) / std)
    for k in original:
        for stat in original[k]:
            torch.testing.assert_close(original[k][stat], saved[k][stat])
    expected = policy.predict_action_chunk(processed)
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)

    def no_download(*args, **kwargs):
        raise AssertionError("Network access during checkpoint restore")

    monkeypatch.setattr(Dinov2Model, "from_pretrained", no_download)
    import socket

    monkeypatch.setattr(socket.socket, "connect", no_download)
    restored = ACTPolicy.from_pretrained(tmp_path, local_files_only=True, strict=True)
    pre2, post2 = make_pre_post_processors(restored.config, pretrained_path=tmp_path)
    actual = restored.predict_action_chunk(pre2(deepcopy(data)))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(post2(actual), post(expected))
    restored.train()
    assert not restored.model.backbone.model.training
    assert all(not p.requires_grad for p in restored.model.backbone.parameters())


def test_bad_image_shape(config):
    policy = ACTPolicy(config)
    with pytest.raises(ValueError, match="14"):
        policy.model.backbone(torch.rand(1, 3, 27, 28))
    with pytest.raises(ValueError, match="RGB"):
        policy.model.backbone(torch.rand(1, 1, 28, 28))


def test_unsupported_backbone():
    with pytest.raises(ValueError):
        ACTConfig(vision_backbone="unknown")


def test_legacy_resnet_defaults():
    assert ACTConfig(device="cpu").vision_backbone == "resnet18"


def test_resnet_roundtrip_preserves_behavior(config, tmp_path):
    config.vision_backbone = "resnet18"
    config.pretrained_backbone_weights = None
    policy = ACTPolicy(config).eval()
    data = batch(config)
    with torch.no_grad():
        expected = policy.predict_action_chunk(data)
    pre, _ = make_act_pre_post_processors(config, stats(config))
    torch.testing.assert_close(
        pre(deepcopy(data))["observation.images.front"], data["observation.images.front"]
    )
    policy.save_pretrained(tmp_path)
    restored = ACTPolicy.from_pretrained(tmp_path, local_files_only=True, strict=True)
    torch.testing.assert_close(restored.predict_action_chunk(data), expected)


def test_missing_dinov2_dependency(config, monkeypatch):
    from lerobot.utils import import_utils

    monkeypatch.setitem(import_utils._require_package_cache, "transformers", False)
    with pytest.raises(ImportError, match="transformers-dep"):
        ACTPolicy(config)
