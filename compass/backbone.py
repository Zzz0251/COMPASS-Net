from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((x, skip), dim=1))


class DINOv3ResUNet(nn.Module):
    """DINOv3 ViT-B/16 encoder with a ResUNet-style dual-head decoder."""

    def __init__(
        self,
        model_path: str | Path,
        image_size: int = 224,
        encoder_layers: Sequence[int] = (2, 5, 8, 11),
        freeze_backbone: bool = False,
        deep_supervision: bool = True,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.encoder_layers = tuple(encoder_layers)
        self.deep_supervision = deep_supervision
        self.encoder = AutoModel.from_pretrained(str(model_path), local_files_only=local_files_only)

        config = self.encoder.config
        hidden_size = int(config.hidden_size)
        self.patch_size = int(getattr(config, "patch_size", 16))
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 4))
        self.num_prefix_tokens = 1 + self.num_register_tokens
        if image_size % self.patch_size:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={self.patch_size}")
        if max(self.encoder_layers) >= int(config.num_hidden_layers):
            raise ValueError(f"Requested encoder layer {max(self.encoder_layers)} from a {config.num_hidden_layers}-layer model")

        for parameter in self.encoder.parameters():
            parameter.requires_grad = not freeze_backbone

        self.input_stem = ResidualConvBlock(3, 32)
        self.skip_112 = nn.Conv2d(hidden_size, 64, 1)
        self.skip_56 = nn.Conv2d(hidden_size, 128, 1)
        self.skip_28 = nn.Conv2d(hidden_size, 256, 1)
        self.bottleneck = ResidualConvBlock(hidden_size, 512)

        self.up_28 = UpBlock(512, 256, 256)
        self.up_56 = UpBlock(256, 128, 128)
        self.up_112 = UpBlock(128, 64, 64)
        self.up_224 = UpBlock(64, 32, 64)
        self.infarct_head = nn.Conv2d(64, 1, 1)
        self.ischemic_head = nn.Conv2d(64, 1, 1)

        if deep_supervision:
            self.aux_28 = nn.Conv2d(256, 2, 1)
            self.aux_56 = nn.Conv2d(128, 2, 1)

        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("encoder."):
                yield parameter

    def _prepare_ncct(self, ncct: torch.Tensor) -> torch.Tensor:
        if ncct.ndim != 4:
            raise ValueError(f"Expected BCHW NCCT input, got shape {tuple(ncct.shape)}")
        if ncct.shape[-2:] != (self.image_size, self.image_size):
            ncct = F.interpolate(ncct, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if ncct.shape[1] == 1:
            ncct = ncct.repeat(1, 3, 1, 1)
        elif ncct.shape[1] != 3:
            raise ValueError(f"Expected one or three input channels, got {ncct.shape[1]}")
        ncct = ncct.clamp(0.0, 1.0)
        return (ncct - self.image_mean) / self.image_std

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        grid = self.image_size // self.patch_size
        expected = grid * grid
        if patch_tokens.shape[1] != expected:
            inferred = int(math.sqrt(patch_tokens.shape[1]))
            if inferred * inferred != patch_tokens.shape[1]:
                raise RuntimeError(
                    f"Cannot reshape {patch_tokens.shape[1]} DINOv3 patch tokens into a square feature map"
                )
            grid = inferred
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid, grid)

    def forward(self, ncct: torch.Tensor) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        original_size = ncct.shape[-2:]
        pixels = self._prepare_ncct(ncct)
        outputs = self.encoder(pixel_values=pixels, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        # hidden_states[0] is the embedding output; +1 maps zero-based transformer layer IDs.
        features = [self._tokens_to_map(hidden_states[index + 1]) for index in self.encoder_layers]

        input_skip = self.input_stem(pixels)
        skip_112 = F.interpolate(self.skip_112(features[0]), scale_factor=8, mode="bilinear", align_corners=False)
        skip_56 = F.interpolate(self.skip_56(features[1]), scale_factor=4, mode="bilinear", align_corners=False)
        skip_28 = F.interpolate(self.skip_28(features[2]), scale_factor=2, mode="bilinear", align_corners=False)
        x = self.bottleneck(features[3])

        x_28 = self.up_28(x, skip_28)
        x_56 = self.up_56(x_28, skip_56)
        x_112 = self.up_112(x_56, skip_112)
        decoded = self.up_224(x_112, input_skip)
        logits = torch.cat((self.infarct_head(decoded), self.ischemic_head(decoded)), dim=1)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, original_size, mode="bilinear", align_corners=False)

        result: Dict[str, torch.Tensor | list[torch.Tensor]] = {
            "logits": logits,
            "features": F.adaptive_avg_pool2d(decoded, 1).flatten(1),
        }
        if self.deep_supervision:
            result["aux_logits"] = [
                F.interpolate(self.aux_28(x_28), original_size, mode="bilinear", align_corners=False),
                F.interpolate(self.aux_56(x_56), original_size, mode="bilinear", align_corners=False),
            ]
        else:
            result["aux_logits"] = []
        return result


class RatioClassifier(nn.Module):
    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features).squeeze(-1)


class SegmentationAttention(nn.Module):
    """Original ResNet attention rule, with channel 1 interpreted as ischemic tissue."""

    def forward(self, logits: torch.Tensor, brain_mask: torch.Tensor | None = None) -> torch.Tensor:
        infarct = torch.sigmoid(logits[:, 0:1])
        ischemic = torch.sigmoid(logits[:, 1:2])
        attention = (infarct + 0.6 * ischemic + 0.2).clamp(0.0, 1.0)
        if brain_mask is not None:
            if brain_mask.shape[-2:] != attention.shape[-2:]:
                brain_mask = F.interpolate(brain_mask, attention.shape[-2:], mode="nearest")
            attention = attention * brain_mask
        return attention


class CoMPASSNet(nn.Module):
    def __init__(
        self,
        dinov3_model_path: str | Path,
        image_size: int = 224,
        freeze_backbone: bool = False,
        deep_supervision: bool = True,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.segmentor = DINOv3ResUNet(
            model_path=dinov3_model_path,
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            deep_supervision=deep_supervision,
            local_files_only=local_files_only,
        )
        self.ratio_classifier = RatioClassifier(feature_dim=64)
        self.attention = SegmentationAttention()

    def forward(self, ncct: torch.Tensor, brain_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        output = self.segmentor(ncct)
        logits = output["logits"]
        features = output["features"]
        assert isinstance(logits, torch.Tensor) and isinstance(features, torch.Tensor)
        ratio_logit = self.ratio_classifier(features)
        output["ratio_logit"] = ratio_logit
        output["ratio_probability"] = torch.sigmoid(ratio_logit)
        output["attention_map"] = self.attention(logits, brain_mask)
        return output
