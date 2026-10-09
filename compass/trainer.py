from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback so training/evaluation remain usable without TensorBoard."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            print("Warning: tensorboard is not installed; scalar logging is disabled.")

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            return None

        def close(self) -> None:
            return None

from .backbone import CoMPASSNet
from .data import create_dataloader
from .diffusion import DiffusionUNet, GaussianDiffusion
from .losses import (
    binary_dice,
    mismatch_consistency_loss,
    mismatch_labels,
    segmentation_loss,
)


@dataclass
class ExperimentConfig:
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    num_workers: int = 4
    ratio_threshold: float = 1.8
    ratio_loss_weight: float = 0.2
    segmentation_weight_start: float = 1.0
    segmentation_weight_end: float = 0.5
    diffusion_weight_start: float = 0.5
    diffusion_weight_end: float = 1.0
    diffusion_timesteps: int = 1000
    sampling_timesteps: int = 100
    save_every: int = 2020
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        return cls(**values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MainExperimentTrainer:
    def __init__(
        self,
        data_root: str | Path,
        dinov3_model_path: str | Path,
        output_dir: str | Path,
        config: ExperimentConfig,
        device: str = "cuda",
        freeze_backbone: bool = False,
        local_files_only: bool = True,
        use_amp: bool = True,
    ) -> None:
        seed_everything(config.seed)
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        self.train_loader = create_dataloader(
            data_root,
            split="train",
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_workers=config.num_workers,
            require_tmax=True,
        )
        test_root = Path(data_root) / "test"
        self.test_loader = (
            create_dataloader(
                data_root,
                split="test",
                batch_size=1,
                image_size=config.image_size,
                num_workers=config.num_workers,
                require_tmax=False,
            )
            if test_root.exists()
            else None
        )

        self.model = CoMPASSNet(
            dinov3_model_path=dinov3_model_path,
            image_size=config.image_size,
            freeze_backbone=freeze_backbone,
            deep_supervision=True,
            local_files_only=local_files_only,
        ).to(self.device)
        self.denoiser = DiffusionUNet(base_dim=64, self_condition=True).to(self.device)
        self.diffusion = GaussianDiffusion(
            self.denoiser,
            image_size=config.image_size,
            timesteps=config.diffusion_timesteps,
            sampling_timesteps=config.sampling_timesteps,
            self_condition_probability=0.5,
        ).to(self.device)

        optimizer_groups = [
            {
                "params": [p for p in self.model.segmentor.backbone_parameters() if p.requires_grad],
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": list(self.model.segmentor.decoder_parameters()),
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": list(self.model.ratio_classifier.parameters()),
                "lr": config.learning_rate * 1.2,
                "weight_decay": config.weight_decay,
            },
            {
                "params": list(self.denoiser.parameters()),
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
        ]
        optimizer_groups = [group for group in optimizer_groups if group["params"]]
        self.optimizer = AdamW(optimizer_groups)
        steps_per_epoch = max(1, len(self.train_loader))
        warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)

        def learning_rate_multiplier(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return 0.95 ** (step // (steps_per_epoch * 5))

        self.scheduler = LambdaLR(self.optimizer, learning_rate_multiplier)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.writer = SummaryWriter(str(self.log_dir))
        self.global_step = 0
        self.start_epoch = 0
        self.best_mean_dice = -1.0

        with (self.output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _loss_weights(self, epoch: int) -> tuple[float, float]:
        progress = epoch / max(self.config.epochs, 1)
        segmentation = self.config.segmentation_weight_start + progress * (
            self.config.segmentation_weight_end - self.config.segmentation_weight_start
        )
        diffusion = self.config.diffusion_weight_start + progress * (
            self.config.diffusion_weight_end - self.config.diffusion_weight_start
        )
        return segmentation, diffusion

    def _segmentation_strength(self, epoch: int) -> float:
        return min(1.0, 0.3 + 0.7 * epoch / max(self.config.epochs, 1))

    def compute_losses(self, batch: Dict[str, Any], epoch: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ncct = batch["ncct"]
        infarct = batch["infarct"]
        ischemic = batch["ischemic"]
        brain_mask = batch["brain_mask"]
        tmax = batch["tmax"]

        output = self.model(ncct, brain_mask)
        logits = output["logits"]
        aux_logits = output["aux_logits"]
        ratio_probability = output["ratio_probability"]
        ratio_logit = output["ratio_logit"]
        attention = output["attention_map"]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(aux_logits, list)
        assert isinstance(ratio_probability, torch.Tensor)
        assert isinstance(ratio_logit, torch.Tensor)
        assert isinstance(attention, torch.Tensor)

        seg_loss, seg_parts = segmentation_loss(logits, aux_logits, infarct, ischemic)
        ratio_label, _ = mismatch_labels(infarct, ischemic, self.config.ratio_threshold)
        ratio_classification = F.binary_cross_entropy_with_logits(ratio_logit, ratio_label)
        ratio_consistency = mismatch_consistency_loss(
            logits[:, 0:1], logits[:, 1:2], ratio_label, self.config.ratio_threshold
        )
        ratio_loss = ratio_classification + 0.5 * ratio_consistency

        lower, upper = self.diffusion.training_timestep_range(epoch, self.config.epochs)
        timestep = torch.randint(lower, upper, (ncct.shape[0],), device=self.device)
        strength = self._segmentation_strength(epoch)
        diffusion_loss = self.diffusion.training_loss(tmax, attention, brain_mask, timestep, strength)
        segmentation_weight, diffusion_weight = self._loss_weights(epoch)
        total = (
            segmentation_weight * seg_loss
            + diffusion_weight * diffusion_loss
            + self.config.ratio_loss_weight * ratio_loss
        )
        parts = {
            "total": total,
            "segmentation": seg_loss,
            "diffusion": diffusion_loss,
            "mismatch": ratio_loss,
            "infarct": seg_parts["infarct"],
            "ischemic": seg_parts["ischemic"],
            "auxiliary": seg_parts["auxiliary"],
            "segmentation_weight": total.new_tensor(segmentation_weight),
            "diffusion_weight": total.new_tensor(diffusion_weight),
            "segmentation_strength": total.new_tensor(strength),
        }
        return total, parts

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.denoiser.train()
        totals: Dict[str, list[float]] = {}
        progress = tqdm(self.train_loader, desc=f"train {epoch + 1}/{self.config.epochs}")
        for raw_batch in progress:
            batch = self._to_device(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, parts = self.compute_losses(batch, epoch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.denoiser.parameters()), max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for key, value in parts.items():
                scalar = float(value.detach().cpu())
                totals.setdefault(key, []).append(scalar)
                self.writer.add_scalar(f"train/{key}", scalar, self.global_step)
            self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            progress.set_postfix(loss=f"{totals['total'][-1]:.4f}")
            self.global_step += 1
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(self.checkpoint_dir / f"step_{self.global_step}.pt", epoch)
        return {key: float(np.mean(values)) for key, values in totals.items()}

    @torch.no_grad()
    def validate(self) -> Optional[Dict[str, float]]:
        if self.test_loader is None:
            return None
        self.model.eval()
        scores = {"infarct_dice": [], "ischemic_dice": []}
        for raw_batch in tqdm(self.test_loader, desc="validation", leave=False):
            batch = self._to_device(raw_batch)
            output = self.model(batch["ncct"], batch["brain_mask"])
            logits = output["logits"]
            assert isinstance(logits, torch.Tensor)
            prediction = torch.sigmoid(logits) >= 0.5
            scores["infarct_dice"].append(float(binary_dice(prediction[:, 0:1], batch["infarct"]).cpu()))
            scores["ischemic_dice"].append(float(binary_dice(prediction[:, 1:2], batch["ischemic"]).cpu()))
        return {key: float(np.mean(values)) for key, values in scores.items()}

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.config.epochs):
            train_metrics = self.train_one_epoch(epoch)
            validation = self.validate()
            summary = {f"train_{key}": value for key, value in train_metrics.items()}
            if validation is not None:
                summary.update(validation)
                mean_dice = 0.5 * (validation["infarct_dice"] + validation["ischemic_dice"])
                if mean_dice > self.best_mean_dice:
                    self.best_mean_dice = mean_dice
                    self.save_checkpoint(self.checkpoint_dir / "best.pt", epoch)
                for key, value in validation.items():
                    self.writer.add_scalar(f"validation/{key}", value, epoch)
            print(json.dumps({"epoch": epoch + 1, **summary}, sort_keys=True))
            self.save_checkpoint(self.checkpoint_dir / "last.pt", epoch)
        self.writer.close()

    @torch.no_grad()
    def dry_run(self) -> Dict[str, float]:
        self.model.train()
        self.denoiser.train()
        batch = self._to_device(next(iter(self.train_loader)))
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            _, parts = self.compute_losses(batch, epoch=0)
        return {key: float(value.detach().cpu()) for key, value in parts.items()}

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_mean_dice": self.best_mean_dice,
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "denoiser": self.denoiser.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            Path(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.denoiser.load_state_dict(checkpoint["denoiser"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_mean_dice = float(checkpoint.get("best_mean_dice", -1.0))
