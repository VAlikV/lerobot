"""Unit tests for Classifier monkey-patch (idempotent, kwargs-tolerant)."""

from __future__ import annotations

import importlib

import pytest


def _reset_patch_flag():
    # Allow re-applying patch in same process (each test imports fresh).
    mod = importlib.import_module("lerobot.processor.reward_model._classifier_patch")
    mod._PATCHED = False  # type: ignore[attr-defined]


@pytest.fixture()
def classifier_class():
    # Keep the patched state across tests but verify behaviour deterministically.
    from lerobot.processor.reward_model import _classifier_patch  # noqa: F401

    try:
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
    except ImportError:
        pytest.skip("Classifier not importable")
    return Classifier


def test_patch_is_idempotent(classifier_class):
    """Re-applying the patch should not double-wrap predict_reward."""
    from lerobot.processor.reward_model._classifier_patch import apply

    first = classifier_class.predict_reward
    apply()
    second = classifier_class.predict_reward
    assert first is second


def test_init_tolerates_extra_kwargs(classifier_class):
    """Patched __init__ must accept (and ignore) extra kwargs like dataset_stats."""
    from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

    cfg = RewardClassifierConfig()
    # Real ctor requires input_features, but even stub-level validation shows
    # the patched init accepts extra kwargs without TypeError.
    try:
        classifier_class(cfg, dataset_stats={"foo": "bar"}, something_else=42)
    except TypeError as e:
        # Should NOT be "got unexpected keyword argument".
        assert "unexpected keyword" not in str(e), e
    except Exception:
        # Any non-TypeError (shape/config issues) is OK — we only care that
        # kwargs don't cause "unexpected keyword" errors.
        pass
