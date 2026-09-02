import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.extended_act.configuration_act import ACTConfig, OBS_GEOMETRY
from lerobot.policies.extended_act.modeling_act import ACTPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import ACTION, OBS_ENV_STATE


def make_config(*, use_vae: bool = True) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
            OBS_GEOMETRY: PolicyFeature(type=FeatureType.STATE, shape=(2, 6)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
        device="cpu",
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        dropout=0.0,
        use_vae=use_vae,
    )


def make_batch() -> dict[str, torch.Tensor]:
    return {
        OBS_ENV_STATE: torch.randn(2, 4),
        OBS_GEOMETRY: torch.randn(2, 2, 6),
        ACTION: torch.randn(2, 4, 3),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }


def test_geometry_is_used_during_training() -> None:
    policy = ACTPolicy(make_config())
    policy.train()

    loss, loss_dict = policy(make_batch())
    loss.backward()

    projection_grad = policy.model.encoder_geometry_input_proj.weight.grad
    assert loss.ndim == 0
    assert set(loss_dict) == {"l1_loss", "kld_loss"}
    assert projection_grad is not None
    assert torch.count_nonzero(projection_grad) > 0


def test_geometry_changes_inference_output() -> None:
    policy = ACTPolicy(make_config(use_vae=False)).eval()
    batch = make_batch()
    batch[OBS_GEOMETRY] = torch.zeros_like(batch[OBS_GEOMETRY])

    with torch.no_grad():
        actions_without_geometry = policy.predict_action_chunk(batch)
        batch[OBS_GEOMETRY] = torch.ones_like(batch[OBS_GEOMETRY])
        actions_with_geometry = policy.predict_action_chunk(batch)

    assert actions_with_geometry.shape == (2, 4, 3)
    assert not torch.equal(actions_without_geometry, actions_with_geometry)


def test_geometry_shape_is_validated() -> None:
    policy = ACTPolicy(make_config(use_vae=False)).eval()
    batch = make_batch()
    batch[OBS_GEOMETRY] = torch.randn(2, 12)

    with pytest.raises(ValueError, match="observation.geometry"):
        policy.predict_action_chunk(batch)


def test_extended_act_is_available_through_dynamic_factory() -> None:
    assert make_policy_config("extended_act").type == "extended_act"
    assert get_policy_class("extended_act") is ACTPolicy
