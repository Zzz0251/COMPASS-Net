import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AdapterModule(nn.Module):
    def __init__(self, dim, reduction_factor=16, adapter_type="bottleneck"):
        super().__init__()
        self.adapter_type = adapter_type
        hidden_dim = max(dim // reduction_factor, 1)

        if adapter_type == "bottleneck":
            self.down_proj = nn.Linear(dim, hidden_dim)
            self.up_proj = nn.Linear(hidden_dim, dim)
            self.act = nn.GELU()
        elif adapter_type == "lora":
            self.lora_a = nn.Linear(dim, hidden_dim, bias=False)
            self.lora_b = nn.Linear(hidden_dim, dim, bias=False)
            self.scaling = 0.1
        else:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}")

    def forward(self, x):
        if self.adapter_type == "bottleneck":
            return x + self.up_proj(self.act(self.down_proj(x)))
        return x + self.lora_b(self.lora_a(x)) * self.scaling


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))


class ResUNetUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DINOv3SegmentationUNet(nn.Module):
                                                                                  

    def __init__(
        self,
        dinov3_model_path,
        output_channels=2,
        image_size=224,
        freeze_backbone=False,
        deep_supervision=True,
        adapter_config=None,
        use_adapters=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.deep_supervision = deep_supervision
        self.use_adapters = use_adapters

        self.dinov3 = AutoModel.from_pretrained(dinov3_model_path, local_files_only=True)
        self.feature_dim = self.dinov3.config.hidden_size
        self.patch_size = getattr(self.dinov3.config, "patch_size", 16)
        self.num_class_tokens = 1
        self.num_register_tokens = getattr(self.dinov3.config, "num_register_tokens", 4)
        self.skip_layer_indices = (1, 3, 6, 9)

        for param in self.dinov3.parameters():
            param.requires_grad = not freeze_backbone

        adapter_config = adapter_config or {
            "type": "bottleneck",
            "reduction_factor": 16,
            "insert_layers": [3, 6, 9, 11],
        }
        self.adapters = nn.ModuleDict()
        if use_adapters:
            for layer_idx in adapter_config["insert_layers"]:
                self.adapters[f"layer_{layer_idx}"] = AdapterModule(
                    self.feature_dim,
                    reduction_factor=adapter_config.get("reduction_factor", 16),
                    adapter_type=adapter_config.get("type", "bottleneck"),
                )

        self.bottleneck = ResidualConvBlock(self.feature_dim, 512)
        self.skip_projs = nn.ModuleDict(
            {
                "s1": nn.Conv2d(self.feature_dim, 64, kernel_size=1),
                "s3": nn.Conv2d(self.feature_dim, 128, kernel_size=1),
                "s6": nn.Conv2d(self.feature_dim, 256, kernel_size=1),
                "s9": nn.Conv2d(self.feature_dim, 256, kernel_size=1),
            }
        )
        self.up1 = ResUNetUpBlock(512, 256, 256)
        self.up2 = ResUNetUpBlock(256, 256, 128)
        self.up3 = ResUNetUpBlock(128, 128, 64)
        self.up4 = ResUNetUpBlock(64, 64, 64)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [nn.Conv2d(256, output_channels, 1), nn.Conv2d(128, output_channels, 1)]
            )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def get_backbone_parameters(self):
        return self.dinov3.parameters()

    def get_adapter_parameters(self):
        params = []
        for adapter in self.adapters.values():
            params.extend(adapter.parameters())
        return params

    def get_decoder_parameters(self):
        decoder = [self.bottleneck, self.skip_projs, self.up1, self.up2, self.up3, self.up4, self.final_conv]
        if self.deep_supervision:
            decoder.append(self.aux_heads)
        params = []
        for module in decoder:
            params.extend(module.parameters())
        return params

    def _prepare_input(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3]
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-6)
        return (x - self.imagenet_mean) / self.imagenet_std

    def _hidden_states_with_adapters(self, x):
        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if not self.use_adapters:
            return hidden_states

        adapted = []
        for idx, hidden_state in enumerate(hidden_states):
            key = f"layer_{idx}"
            adapted.append(self.adapters[key](hidden_state) if key in self.adapters else hidden_state)
        return tuple(adapted)

    def _tokens_to_feature_map(self, tokens, height, width):
        start = self.num_class_tokens + self.num_register_tokens
        patch_tokens = tokens[:, start:, :]
        expected_h = height // self.patch_size
        expected_w = width // self.patch_size
        expected_tokens = expected_h * expected_w

        if patch_tokens.shape[1] != expected_tokens:
            side = int(math.sqrt(patch_tokens.shape[1]))
            expected_h = expected_w = side
            patch_tokens = patch_tokens[:, : side * side, :]

        bsz, _, dim = patch_tokens.shape
        return patch_tokens.reshape(bsz, expected_h, expected_w, dim).permute(0, 3, 1, 2).contiguous()

    def _feature_at(self, hidden_states, layer_idx):
        idx = min(layer_idx, len(hidden_states) - 1)
        return self._tokens_to_feature_map(hidden_states[idx], self.image_size, self.image_size)

    def forward(self, x, return_features=False):
        original_size = x.shape[-2:]
        x = self._prepare_input(x)
        hidden_states = self._hidden_states_with_adapters(x)

        s1 = self.skip_projs["s1"](self._feature_at(hidden_states, self.skip_layer_indices[0]))
        s3 = self.skip_projs["s3"](self._feature_at(hidden_states, self.skip_layer_indices[1]))
        s6 = self.skip_projs["s6"](self._feature_at(hidden_states, self.skip_layer_indices[2]))
        s9 = self.skip_projs["s9"](self._feature_at(hidden_states, self.skip_layer_indices[3]))

        x = self.bottleneck(self._tokens_to_feature_map(hidden_states[-1], self.image_size, self.image_size))
        x = self.up1(x, s9)
        aux_outputs = []
        if self.deep_supervision:
            aux_outputs.append(F.interpolate(self.aux_heads[0](x), size=original_size, mode="bilinear", align_corners=False))

        x = self.up2(x, s6)
        if self.deep_supervision:
            aux_outputs.append(F.interpolate(self.aux_heads[1](x), size=original_size, mode="bilinear", align_corners=False))

        x = self.up3(x, s3)
        x = self.up4(x, s1)
        logits = self.final_conv(x)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)

        pooled_features = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if return_features:
            if self.deep_supervision:
                return logits, pooled_features, aux_outputs
            return logits, pooled_features
        return logits


class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, use_boundary_aware=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, segmentation_pred, brain_mask=None):
        infarct_prob = torch.sigmoid(segmentation_pred[:, 0:1])
        penumbra_prob = torch.sigmoid(segmentation_pred[:, 1:2])
        attention_map = infarct_prob + 0.6 * penumbra_prob + 0.2
        attention_map = attention_map.clamp(0, 1)
        if brain_mask is not None:
            if brain_mask.shape[-2:] != attention_map.shape[-2:]:
                brain_mask = F.interpolate(brain_mask, size=attention_map.shape[-2:], mode="nearest")
            attention_map = attention_map * brain_mask
        return attention_map


def default_dinov3_path():
    here = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(here, " ", " ", " ")
    if os.path.isdir(local_path):
        return os.path.abspath(local_path)
    return os.path.abspath(os.path.join(here, "..", " ", " ", " "))
