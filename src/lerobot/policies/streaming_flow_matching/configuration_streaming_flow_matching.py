"""Configuration and named state/action contract for Streaming Flow Policy."""

import math
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("streaming_flow_matching")
@dataclass
class StreamingFlowMatchingConfig(PreTrainedConfig):
    """State-imitation SFP. Time [0, 1] spans ``horizon - 1`` dataset intervals.

    ``n_action_steps`` controls when observations/initial measured pose are refreshed;
    actions are integrated and returned individually even when it is greater than one.
    Names come from dataset metadata, and remain in the checkpoint for deployment.
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    variant: str = "deterministic"
    sigma: float = 0.1
    sigma1: float = 0.1
    stabilizing_gain: float = 0.0
    integration_steps_per_action: int = 1
    state_feature_names: list[str] | None = None
    action_feature_names: list[str] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    down_dims: tuple[int, ...] = (256, 512, 1024)
    n_groups: int = 8
    kernel_size: int = 5
    time_embed_dim: int = 256
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    vision_feature_dim: int = 64
    resize_shape: tuple[int, int] = (224, 224)
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in ("deterministic", "stochastic"):
            raise ValueError("variant must be deterministic or stochastic")
        if self.horizon < 2 or not 1 <= self.n_action_steps < self.horizon:
            raise ValueError("Require horizon >= 2 and 1 <= n_action_steps < horizon")
        if self.n_obs_steps < 1 or self.integration_steps_per_action < 1:
            raise ValueError("n_obs_steps and integration_steps_per_action must be positive")
        if not all(math.isfinite(x) and x >= 0 for x in (self.sigma, self.sigma1, self.stabilizing_gain)):
            raise ValueError("Flow noise and stabilizing_gain must be finite and non-negative")
        if self.variant == "stochastic" and self.sigma1 < self.sigma:
            raise ValueError("stochastic requires sigma1 >= sigma")
        if self.variant == "stochastic" and self.stabilizing_gain:
            raise ValueError("stabilizing_gain applies only to the deterministic flow")
        if (
            not self.down_dims
            or self.n_groups < 1
            or any(d < self.n_groups or d % self.n_groups for d in self.down_dims)
        ):
            raise ValueError("down_dims must be positive multiples of n_groups")
        if self.time_embed_dim < 4 or self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be even and >= 4")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        if self.vision_backbone != "resnet18" or self.vision_feature_dim < 1:
            raise ValueError("Use resnet18 with a positive vision_feature_dim")
        if len(self.resize_shape) != 2 or min(self.resize_shape) < 1:
            raise ValueError("resize_shape must contain two positive dimensions")

    @property
    def drop_n_last_frames(self) -> int:
        # The final frame has no measured successor and therefore no derivative.
        return 1

    @property
    def eval_drop_n_last_frames(self) -> int:
        return self.drop_n_last_frames

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def state_observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, self.horizon))

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def state_action_indices(self) -> list[int]:
        if not self.state_feature_names or not self.action_feature_names:
            raise ValueError("State and action feature names are required from dataset metadata")
        if len(set(self.state_feature_names)) != len(self.state_feature_names) or len(
            set(self.action_feature_names)
        ) != len(self.action_feature_names):
            raise ValueError("State and action feature names must be unique")
        missing = set(self.action_feature_names) - set(self.state_feature_names)
        if missing:
            raise ValueError(f"Measured state is missing action features: {sorted(missing)}")
        return [self.state_feature_names.index(name) for name in self.action_feature_names]

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        state_names = features.get(OBS_STATE, {}).get("names")
        action_names = features.get(ACTION, {}).get("names")
        if not state_names or not action_names:
            raise ValueError("Dataset must provide named observation.state and action features")
        # A loaded policy cannot silently reinterpret its learned feature ordering.
        if self.pretrained_path and (
            (self.state_feature_names and self.state_feature_names != list(state_names))
            or (self.action_feature_names and self.action_feature_names != list(action_names))
        ):
            raise ValueError("Dataset feature order differs from the pretrained SFP contract")
        self.state_feature_names = list(state_names)
        self.action_feature_names = list(action_names)
        _ = self.state_action_indices

    def validate_dataset_metadata(self, metadata: Any, episodes: list[int] | None = None) -> None:
        if metadata.total_frames == 0 or metadata.total_episodes == 0:
            raise ValueError("Cannot train SFP on an empty dataset")
        self.set_dataset_feature_metadata(metadata.features)
        # No valid velocity can be learned from a collection of singleton episodes.
        selected = episodes if episodes is not None else range(metadata.total_episodes)
        if not any(metadata.episodes[i]["length"] >= 2 for i in selected):
            raise ValueError("Dataset has no valid two-frame trajectory intervals")

    def validate_features(self) -> None:
        if self.robot_state_feature is None or self.action_feature is None:
            raise ValueError("SFP requires observation.state and action features")
        if self.robot_state_feature.shape != (len(self.state_feature_names or []),):
            raise ValueError("state_feature_names must match observation.state dimension")
        if self.action_feature.shape != (len(self.action_feature_names or []),):
            raise ValueError("action_feature_names must match action dimension")
        _ = self.state_action_indices

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> CosineAnnealingWithWarmupSchedulerConfig:
        return CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)
