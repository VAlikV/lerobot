"""Split measured future targets from observation history before normalization."""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_streaming_flow_matching import StreamingFlowMatchingConfig

ANCHOR = "sfp_anchor"
VALID_INTERVALS = "sfp_valid_intervals"


def measured_action_stats(config: StreamingFlowMatchingConfig, stats: dict) -> dict:
    """Use measured-state statistics in action order, never teleoperator-command stats."""
    if OBS_STATE not in stats:
        raise ValueError("Measured observation.state statistics are required for SFP")
    indices = config.state_action_indices
    action_stats = {}
    for name, value in stats[OBS_STATE].items():
        value = torch.as_tensor(value)
        action_stats[name] = (
            value[indices].clone()
            if value.ndim and value.numel() == len(config.state_feature_names)
            else value.clone()
        )
    if not {"min", "max"} <= action_stats.keys():
        raise ValueError("Measured state min/max statistics are required for SFP")
    return {**stats, ACTION: action_stats}


@dataclass
@ProcessorStepRegistry.register("sfp_measured_trajectory")
class MeasuredTrajectoryProcessorStep(ProcessorStep):
    state_action_indices: list[int]
    n_obs_steps: int
    horizon: int

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        obs = dict(result[TransitionKey.OBSERVATION])
        comp = dict(result.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        state = obs[OBS_STATE]
        if state.ndim == 3 and state.shape[1] == self.n_obs_steps + self.horizon - 1:
            # Training: [-history, ..., current, ..., future].
            future = state[:, self.n_obs_steps - 1 :]
            result[TransitionKey.ACTION] = future[..., self.state_action_indices].clone()
            pad = obs.pop(f"{OBS_STATE}_is_pad", None)
            if pad is None:
                valid = torch.ones(future.shape[:2], device=state.device, dtype=torch.bool)
            else:
                valid = ~pad[:, self.n_obs_steps - 1 :].bool()
            comp[VALID_INTERVALS] = valid[:, :-1] & valid[:, 1:]
            obs[OBS_STATE] = state[:, : self.n_obs_steps].clone()
        elif state.ndim == 2 or (state.ndim == 3 and state.shape[1] == self.n_obs_steps):
            current = state if state.ndim == 2 else state[:, -1]
            result[TransitionKey.ACTION] = current[:, self.state_action_indices].clone()
        else:
            raise ValueError(f"Unexpected SFP state window shape: {tuple(state.shape)}")
        result[TransitionKey.OBSERVATION] = obs
        result[TransitionKey.COMPLEMENTARY_DATA] = comp
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "state_action_indices": self.state_action_indices,
            "n_obs_steps": self.n_obs_steps,
            "horizon": self.horizon,
        }

    def transform_features(self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]) -> dict:
        return features


@dataclass
@ProcessorStepRegistry.register("sfp_normalized_anchor")
class NormalizedAnchorProcessorStep(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        action = result[TransitionKey.ACTION]
        comp = dict(result.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp[ANCHOR] = action[:, 0].clone() if action.ndim == 3 else action.clone()
        result[TransitionKey.COMPLEMENTARY_DATA] = comp
        return result

    def transform_features(self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]) -> dict:
        return features


def make_streaming_flow_matching_pre_post_processors(
    config: StreamingFlowMatchingConfig,
    dataset_stats: dict | None = None,
    dataset_meta: Any | None = None,
):
    if dataset_meta is not None:
        config.set_dataset_feature_metadata(dataset_meta.features)
    if dataset_stats is None:
        raise ValueError("SFP processors require measured dataset statistics")
    steps = make_default_policy_processor_steps(config, measured_action_stats(config, dataset_stats))
    return make_policy_processor_pipelines(
        [
            steps.rename_observations,
            steps.add_batch_dim,
            steps.to_device,
            MeasuredTrajectoryProcessorStep(config.state_action_indices, config.n_obs_steps, config.horizon),
            steps.normalize,
            NormalizedAnchorProcessorStep(),
        ],
        [steps.unnormalize, steps.to_cpu],
    )


def prepare_streaming_processor_kwargs(config: StreamingFlowMatchingConfig, kwargs: dict) -> dict:
    """Rebase trainer finetune stats overrides; resume keeps the checkpoint's stats.

    Importing this module also registers custom steps before loading a checkpoint
    in a new process, without forcing modeling imports into the policy registry.
    """
    result = dict(kwargs)
    for key in ("preprocessor_overrides", "postprocessor_overrides"):
        if key in result:
            overrides = {name: dict(value) for name, value in result[key].items()}
            for name in ("normalizer_processor", "unnormalizer_processor"):
                if name in overrides and overrides[name].get("stats") is not None:
                    overrides[name]["stats"] = measured_action_stats(config, overrides[name]["stats"])
            result[key] = overrides
    return result
