from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).pow(2)
    cumulative = cumulative / cumulative[0]
    return (1.0 - cumulative[1:] / cumulative[:-1]).clamp(0.0, 0.999)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            torch.arange(half, device=time.device, dtype=torch.float32)
            * -(math.log(10000) / max(half - 1, 1))
        )
        embedding = time.float()[:, None] * frequency[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time(time).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.norm1(self.conv1(x))
        hidden = F.silu(hidden * (1.0 + scale) + shift)
        hidden = F.silu(self.norm2(self.conv2(hidden)))
        return hidden + self.shortcut(x)


class SegmentationGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, attention: torch.Tensor, strength: float) -> torch.Tensor:
        attention = F.interpolate(attention, x.shape[-2:], mode="bilinear", align_corners=False)
        encoded = self.encoder(attention)
        gate = self.gate(torch.cat((x, encoded), dim=1))
        return x * (1.0 + self.alpha * gate * strength) + encoded * strength


class DiffusionUNet(nn.Module):
    """No-text conditional denoiser used by the complete ResNet main experiment."""

    def __init__(self, base_dim: int = 64, self_condition: bool = True) -> None:
        super().__init__()
        self.self_condition = self_condition
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(2 if self_condition else 1, base_dim, 7, padding=3)

        self.down1 = TimeResidualBlock(base_dim, base_dim, time_dim)
        self.gate1 = SegmentationGate(base_dim)
        self.ds1 = nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1)
        self.down2 = TimeResidualBlock(base_dim * 2, base_dim * 2, time_dim)
        self.gate2 = SegmentationGate(base_dim * 2)
        self.ds2 = nn.Conv2d(base_dim * 2, base_dim * 4, 4, stride=2, padding=1)
        self.down3 = TimeResidualBlock(base_dim * 4, base_dim * 4, time_dim)
        self.gate3 = SegmentationGate(base_dim * 4)
        self.ds3 = nn.Conv2d(base_dim * 4, base_dim * 8, 4, stride=2, padding=1)

        self.middle1 = TimeResidualBlock(base_dim * 8, base_dim * 8, time_dim)
        self.middle_gate = SegmentationGate(base_dim * 8)
        self.middle2 = TimeResidualBlock(base_dim * 8, base_dim * 8, time_dim)

        self.us3 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 4, stride=2, padding=1)
        self.up3 = TimeResidualBlock(base_dim * 8, base_dim * 4, time_dim)
        self.us2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.up2 = TimeResidualBlock(base_dim * 4, base_dim * 2, time_dim)
        self.us1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.up1 = TimeResidualBlock(base_dim * 2, base_dim, time_dim)
        self.final_gate = SegmentationGate(base_dim)
        self.output = nn.Conv2d(base_dim, 1, 1)

    def forward(
        self,
        noisy_tmax: torch.Tensor,
        attention_map: torch.Tensor,
        time: torch.Tensor,
        self_condition: torch.Tensor | None = None,
        segmentation_strength: float = 1.0,
    ) -> torch.Tensor:
        if self.self_condition:
            self_condition = torch.zeros_like(noisy_tmax) if self_condition is None else self_condition
            noisy_tmax = torch.cat((self_condition, noisy_tmax), dim=1)
        time_embedding = self.time_mlp(time)

        h1 = self.gate1(self.down1(self.input(noisy_tmax), time_embedding), attention_map, segmentation_strength)
        h2 = self.gate2(self.down2(self.ds1(h1), time_embedding), attention_map, segmentation_strength)
        h3 = self.gate3(self.down3(self.ds2(h2), time_embedding), attention_map, segmentation_strength)
        x = self.middle1(self.ds3(h3), time_embedding)
        x = self.middle2(self.middle_gate(x, attention_map, segmentation_strength), time_embedding)

        x = self.up3(torch.cat((self.us3(x), h3), dim=1), time_embedding)
        x = self.up2(torch.cat((self.us2(x), h2), dim=1), time_embedding)
        x = self.up1(torch.cat((self.us1(x), h1), dim=1), time_embedding)
        return self.output(self.final_gate(x, attention_map, segmentation_strength))


@dataclass
class ModelPrediction:
    noise: torch.Tensor
    start: torch.Tensor


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: DiffusionUNet,
        image_size: int = 224,
        timesteps: int = 1000,
        sampling_timesteps: int = 100,
        self_condition_probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.self_condition_probability = self_condition_probability

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = F.pad(cumulative[:-1], (1, 0), value=1.0)
        for name, value in {
            "betas": betas,
            "alphas_cumprod": cumulative,
            "alphas_cumprod_prev": previous,
            "sqrt_alphas_cumprod": cumulative.sqrt(),
            "sqrt_one_minus_alphas_cumprod": (1.0 - cumulative).sqrt(),
            "sqrt_recip_alphas_cumprod": (1.0 / cumulative).sqrt(),
            "sqrt_recipm1_alphas_cumprod": (1.0 / cumulative - 1.0).sqrt(),
        }.items():
            self.register_buffer(name, value.float())

    def q_sample(self, start: torch.Tensor, time: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, time, start.shape) * start + extract(
            self.sqrt_one_minus_alphas_cumprod, time, start.shape
        ) * noise

    def predict_start(self, noisy: torch.Tensor, time: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_alphas_cumprod, time, noisy.shape) * noisy - extract(
            self.sqrt_recipm1_alphas_cumprod, time, noisy.shape
        ) * noise

    def model_predictions(
        self,
        noisy: torch.Tensor,
        attention: torch.Tensor,
        time: torch.Tensor,
        self_condition: torch.Tensor | None = None,
        segmentation_strength: float = 1.0,
    ) -> ModelPrediction:
        predicted_noise = self.model(noisy, attention, time, self_condition, segmentation_strength)
        predicted_start = self.predict_start(noisy, time, predicted_noise).clamp(-1.0, 1.0)
        return ModelPrediction(predicted_noise, predicted_start)

    def training_loss(
        self,
        tmax: torch.Tensor,
        attention: torch.Tensor,
        brain_mask: torch.Tensor,
        time: torch.Tensor,
        segmentation_strength: float = 1.0,
    ) -> torch.Tensor:
        noise = torch.randn_like(tmax)
        noisy = self.q_sample(tmax, time, noise) * brain_mask
        target = noise
        self_condition = None
        if self.model.self_condition and torch.rand((), device=tmax.device) < self.self_condition_probability:
            with torch.no_grad():
                self_condition = self.model_predictions(
                    noisy, attention, time, segmentation_strength=segmentation_strength
                ).start.detach()
        prediction = self.model(noisy, attention, time, self_condition, segmentation_strength)
        loss = F.mse_loss(prediction, target, reduction="none") + 0.1 * F.l1_loss(
            prediction, target, reduction="none"
        )
        loss = loss * brain_mask * (1.0 + 2.0 * attention)
        return loss.mean()

    def training_timestep_range(self, epoch: int, epochs: int) -> tuple[int, int]:
        progress = epoch / max(epochs, 1)
        if progress < 0.3:
            return self.num_timesteps // 2, self.num_timesteps
        if progress < 0.7:
            return self.num_timesteps // 4, self.num_timesteps
        return 0, self.num_timesteps

    @staticmethod
    def adaptive_weights(epoch: int, epochs: int) -> tuple[float, float]:
        progress = epoch / max(epochs, 1)
        return 1.0 - 0.5 * progress, 0.5 + 0.5 * progress

    @torch.no_grad()
    def sample(
        self,
        attention: torch.Tensor,
        brain_mask: torch.Tensor,
        segmentation_strength: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        batch = attention.shape[0]
        image = torch.randn(
            (batch, 1, self.image_size, self.image_size), device=attention.device, dtype=attention.dtype
        ) * brain_mask
        times = torch.linspace(-1, self.num_timesteps - 1, self.sampling_timesteps + 1, device=image.device)
        times = list(reversed(times.long().tolist()))
        iterator = zip(times[:-1], times[1:])
        if show_progress:
            iterator = tqdm(iterator, total=len(times) - 1, desc="diffusion sampling", leave=False)
        self_condition = None
        for current, following in iterator:
            time = torch.full((batch,), current, device=image.device, dtype=torch.long)
            prediction = self.model_predictions(
                image, attention, time, self_condition, segmentation_strength
            )
            self_condition = prediction.start if self.model.self_condition else None
            if following < 0:
                image = prediction.start
                continue
            alpha_next = self.alphas_cumprod[following]
            image = prediction.start * alpha_next.sqrt() + prediction.noise * (1.0 - alpha_next).sqrt()
            image = image * brain_mask
        return image
