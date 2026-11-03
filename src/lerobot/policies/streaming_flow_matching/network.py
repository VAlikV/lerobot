"""FiLM U-Net with fixed token count, following the author's fc_timesteps network.

The sequence contains one action token (SFP-D) or action and latent tokens (SFP-S),
so down/up layers mix flattened features instead of reducing the temporal length.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional
from torchvision.models import ResNet18_Weights, resnet18

from .configuration_streaming_flow_matching import StreamingFlowMatchingConfig


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, kernel: int, groups: int):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel, padding=kernel // 2), nn.GroupNorm(groups, out_dim), nn.Mish()
        )
        self.second = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel, padding=kernel // 2), nn.GroupNorm(groups, out_dim), nn.Mish()
        )
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * out_dim))
        self.skip = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        scale, bias = self.film(conditioning).chunk(2, -1)
        hidden = self.first(x) * scale[..., None] + bias[..., None]
        return self.second(hidden) + self.skip(x)


class LinearTokens(nn.Module):
    def __init__(self, channels: int, tokens: int):
        super().__init__()
        self.linear = nn.Linear(channels * tokens, channels * tokens)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(1)).reshape_as(x)


class StreamingVelocityNet(nn.Module):
    def __init__(self, config: StreamingFlowMatchingConfig, conditioning_dim: int):
        super().__init__()
        self.time_dim = config.time_embed_dim
        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        cond_dim = conditioning_dim + self.time_dim
        action_dim = config.action_feature.shape[0]
        tokens = 1 if config.variant == "deterministic" else 2
        dims = [action_dim, *config.down_dims]
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            self.down.append(
                nn.ModuleList(
                    [
                        ResidualBlock(a, b, cond_dim, config.kernel_size, config.n_groups),
                        ResidualBlock(b, b, cond_dim, config.kernel_size, config.n_groups),
                        LinearTokens(b, tokens) if i < len(config.down_dims) - 1 else nn.Identity(),
                    ]
                )
            )
        self.middle = nn.ModuleList(
            [
                ResidualBlock(dims[-1], dims[-1], cond_dim, config.kernel_size, config.n_groups)
                for _ in range(2)
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResidualBlock(2 * b, a, cond_dim, config.kernel_size, config.n_groups),
                        ResidualBlock(a, a, cond_dim, config.kernel_size, config.n_groups),
                        LinearTokens(a, tokens),
                    ]
                )
                for a, b in reversed(list(zip(config.down_dims[:-1], config.down_dims[1:], strict=True)))
            ]
        )
        self.output = nn.Sequential(
            nn.Conv1d(dims[1], dims[1], config.kernel_size, padding=config.kernel_size // 2),
            nn.GroupNorm(config.n_groups, dims[1]),
            nn.Mish(),
            nn.Conv1d(dims[1], action_dim, 1),
        )

    def forward(self, x: Tensor, time: Tensor, conditioning: Tensor) -> Tensor:
        frequencies = torch.exp(
            torch.arange(self.time_dim // 2, device=x.device) * (-math.log(10000) / (self.time_dim // 2 - 1))
        )
        angles = time[:, None] * frequencies[None]
        embedding = self.time_encoder(torch.cat((angles.sin(), angles.cos()), -1))
        cond = torch.cat((embedding, conditioning), -1)
        hidden, skips = x.transpose(1, 2), []
        for first, second, mix in self.down:
            hidden = second(first(hidden, cond), cond)
            skips.append(hidden)
            hidden = mix(hidden)
        for block in self.middle:
            hidden = block(hidden, cond)
        for first, second, mix in self.up:
            hidden = torch.cat((hidden, skips.pop()), 1)
            hidden = mix(second(first(hidden, cond), cond))
        return self.output(hidden).transpose(1, 2)


class ObservationEncoder(nn.Module):
    def __init__(self, config: StreamingFlowMatchingConfig):
        super().__init__()
        self.config = config
        self.camera_keys = list(config.image_features)
        weights = (
            ResNet18_Weights[config.pretrained_backbone_weights]
            if config.pretrained_backbone_weights
            else None
        )
        self.cameras = nn.ModuleList()
        for _ in self.camera_keys:
            backbone = resnet18(weights=weights)
            # GroupNorm works with small batches and has no running-state side effects.
            self._replace_batch_norm(backbone)
            backbone.fc = nn.Linear(backbone.fc.in_features, config.vision_feature_dim)
            self.cameras.append(backbone)
        self.output_dim = config.n_obs_steps * (
            config.robot_state_feature.shape[0] + len(self.camera_keys) * config.vision_feature_dim
        )

    @classmethod
    def _replace_batch_norm(cls, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                replacement = nn.GroupNorm(min(32, child.num_features), child.num_features)
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
                setattr(module, name, replacement)
            else:
                cls._replace_batch_norm(child)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = [batch["observation.state"].flatten(1)]
        for key, encoder in zip(self.camera_keys, self.cameras, strict=True):
            image = batch[key]
            batch_size, steps = image.shape[:2]
            image = image.flatten(0, 1)
            image = functional.interpolate(
                image, size=self.config.resize_shape, mode="bilinear", align_corners=False
            )
            features.append(encoder(image).reshape(batch_size, steps * self.config.vision_feature_dim))
        return torch.cat(features, -1)
