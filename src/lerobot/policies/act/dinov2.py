"""Frozen DINOv2 patch features with the ACT spatial-backbone interface."""

import torch
from torch import nn

from lerobot.utils.import_utils import require_package

from .configuration_act import ACTConfig


class FrozenDINOv2Backbone(nn.Module):
    def __init__(self, config: ACTConfig, initialize_pretrained: bool = True):
        super().__init__()
        require_package("transformers", "transformers-dep")
        from transformers import Dinov2Config, Dinov2Model

        if initialize_pretrained:
            self.model = Dinov2Model.from_pretrained(config.dinov2_model_name)
            config.dinov2_config = self.model.config.to_dict()
        else:
            if config.dinov2_config is None:
                raise ValueError("DINOv2 checkpoint is missing its saved architecture (dinov2_config)")
            self.model = Dinov2Model(Dinov2Config.from_dict(config.dinov2_config))
        self.patch_size = self.model.config.patch_size
        if self.patch_size != 14 or self.model.config.num_channels != 3:
            raise ValueError("DINOv2 requires RGB images and patch size 14")
        self.out_channels = self.model.config.hidden_size
        self.model.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True):
        # Preserve eval behavior even when the parent ACT enters training mode.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOv2 requires batched RGB images [B, 3, H, W]")
        b, _, h, w = images.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("DINOv2 image height and width must be divisible by 14")
        patches = self.model(pixel_values=images).last_hidden_state[:, 1:]
        return {
            "feature_map": patches.transpose(1, 2).reshape(
                b, self.out_channels, h // self.patch_size, w // self.patch_size
            )
        }
