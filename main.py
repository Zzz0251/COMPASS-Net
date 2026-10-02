import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from dino_segmentation_model import (  
    DINOv3SegmentationUNet,
    EnhancedAttentionModule,
    default_dinov3_path,
)
from models.diffusion_model import DiffusionUNet  
from models.gaussian_diffusion import GaussianDiffusion  


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
    return 1 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def ce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


def combined_seg_loss(pred, target):
    return dice_loss(pred, target) + ce_loss(pred, target)


def compute_ratio_labels(infarct_mask, ischemic_mask, threshold=1.8, smooth=1e-6):
    infarct_areas = infarct_mask.sum(dim=[1, 2, 3])
    ischemic_areas = ischemic_mask.sum(dim=[1, 2, 3])
    ratios = (ischemic_areas + smooth) / (infarct_areas + smooth)
    return (ratios >= threshold).float(), ratios


def ratio_consistency_loss(infarct_pred, ischemic_pred, ratio_labels, threshold=1.8, smooth=1e-6):
    infarct_prob = torch.sigmoid(infarct_pred)
    ischemic_prob = torch.sigmoid(ischemic_pred)
    infarct_areas = infarct_prob.sum(dim=[1, 2, 3])
    ischemic_areas = ischemic_prob.sum(dim=[1, 2, 3])
    pred_ratios = (ischemic_areas + smooth) / (infarct_areas + smooth)
    pred_labels = torch.sigmoid((pred_ratios - threshold) * 5.0)
    return F.binary_cross_entropy(pred_labels, ratio_labels)


class RatioClassifier(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.classifier(features).squeeze(-1)


class MultimodalDINOTrainer:
    def __init__(
        self,
        data_root="",
        dinov3_model_path=None,
        batch_size=2,
        learning_rate=1e-4,
        backbone_lr=1e-5,
        image_size=224,
        device="cuda",
        checkpoint_dir="",
        log_dir="",
        use_deep_supervision=True,
        use_progressive_training=True,
        use_adaptive_loss_weights=True,
        warmup_epochs=10,
        use_ratio_constraint=True,
        ratio_threshold=1.8,
        ratio_loss_weight=0.2,
        freeze_backbone=False,
        build_loaders=True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.image_size = image_size
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_deep_supervision = use_deep_supervision
        self.use_progressive_training = use_progressive_training
        self.use_adaptive_loss_weights = use_adaptive_loss_weights
        self.warmup_epochs = warmup_epochs
        self.use_ratio_constraint = use_ratio_constraint
        self.ratio_threshold = ratio_threshold
        self.ratio_loss_weight = ratio_loss_weight
        self.current_epoch = 0
        self.step = 0

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.train_loader = None
        self.test_loader = None

        dinov3_model_path = dinov3_model_path or default_dinov3_path()
        self.segmentation_model = DINOv3SegmentationUNet(
            dinov3_model_path=dinov3_model_path,
            output_channels=2,
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            deep_supervision=use_deep_supervision,
        ).to(self.device)
        self.attention_module = EnhancedAttentionModule(in_channels=2, out_channels=1).to(self.device)
        self.ratio_classifier = RatioClassifier(feature_dim=64).to(self.device) if use_ratio_constraint else None

        self.diffusion_model = DiffusionUNet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            output_channels=1,
            text_dim=256,
            use_film=False,
        ).to(self.device)
        self.diffusion = GaussianDiffusion(
            model=self.diffusion_model,
            image_size=image_size,
            timesteps=1000,
            sampling_timesteps=100,
            loss_type="hybrid",
            objective="pred_noise",
            beta_schedule="cosine",
            use_perceptual_loss=True,
            perceptual_weight=0.1,
            use_self_conditioning=True,
            self_condition_prob=0.5,
            classifier_free_guidance=False,
            guidance_scale=1.0,
            condition_dropout_prob=0.0,
        ).to(self.device)

        optimizer_params = [
            {"params": self.segmentation_model.get_backbone_parameters(), "lr": backbone_lr, "weight_decay": 0.01},
            {
                "params": self.segmentation_model.get_adapter_parameters(),
                "lr": learning_rate,
                "weight_decay": 0.01,
            },
            {
                "params": self.segmentation_model.get_decoder_parameters(),
                "lr": learning_rate,
                "weight_decay": 0.01,
            },
            {"params": self.attention_module.parameters(), "lr": learning_rate * 1.5, "weight_decay": 0.01},
            {"params": self.diffusion_model.parameters(), "lr": learning_rate, "weight_decay": 0.01},
        ]
        if self.ratio_classifier is not None:
            optimizer_params.append(
                {"params": self.ratio_classifier.parameters(), "lr": learning_rate * 1.2, "weight_decay": 0.01}
            )
        self.optimizer = optim.AdamW(optimizer_params)
        def warmup_lambda(step):
            steps_per_epoch = max(1, len(self.train_loader)) if self.train_loader is not None else 1
            warmup_steps = max(1, self.warmup_epochs * steps_per_epoch)
            if step < warmup_steps:
                return step / warmup_steps
            return 0.95 ** (step // (steps_per_epoch * 5))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, warmup_lambda)
        self.writer = SummaryWriter(log_dir)
        self.seg_weight = 1.0
        self.diff_weight = 0.5

        if build_loaders:
            from data.dataset import create_dataloader

            self.train_loader = create_dataloader(data_root, batch_size=batch_size, image_size=image_size, mode="train")
            self.test_loader = create_dataloader(data_root, batch_size=1, image_size=image_size, mode="test")

    def get_control_params(self, epoch, total_epochs):
        progress = epoch / max(total_epochs, 1)
        if self.use_progressive_training:
            seg_strength = min(1.0, 0.3 + 0.7 * progress)
            min_t, max_t = self.diffusion.get_current_timestep_range(epoch, total_epochs)
        else:
            seg_strength = 1.0
            min_t, max_t = 0, self.diffusion.num_timesteps
        return seg_strength, min_t, max_t

    def adaptive_loss_weights(self, seg_loss, diff_loss, epoch, total_epochs):
        if not self.use_adaptive_loss_weights:
            return self.seg_weight * seg_loss + self.diff_weight * diff_loss
        return self.diffusion.adaptive_loss_weights(seg_loss, diff_loss, epoch, total_epochs)

    def _resize_batch_to_dino(self, batch):
        resized = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() == 4 and value.shape[-2:] != (self.image_size, self.image_size):
                mode = "nearest" if key in {"infarct", "penumbra", "ischemic", "brain_mask"} else "bilinear"
                resized[key] = F.interpolate(
                    value,
                    size=(self.image_size, self.image_size),
                    mode=mode,
                    align_corners=False if mode == "bilinear" else None,
                )
            else:
                resized[key] = value
        return resized

    def train_step(self, batch, epoch=0, total_epochs=1):
        self.optimizer.zero_grad(set_to_none=True)
        batch = self._resize_batch_to_dino(batch)

        ncct = batch["ncct"].to(self.device)
        infarct = batch["infarct"].to(self.device)
        ischemic = (batch["ischemic"] if "ischemic" in batch else batch["penumbra"]).to(self.device)
        brain_mask = batch.get("brain_mask", torch.ones_like(infarct)).to(self.device)
        seg_strength, min_t, max_t = self.get_control_params(epoch, total_epochs)

        ratio_labels = None
        ratio_pred = None
        ratio_loss = torch.tensor(0.0, device=self.device)
        if self.use_ratio_constraint:
            ratio_labels, _ = compute_ratio_labels(infarct, ischemic, self.ratio_threshold)

        if self.use_deep_supervision:
            segmentation_pred, features, aux_outputs = self.segmentation_model(ncct, return_features=True)
        else:
            segmentation_pred, features = self.segmentation_model(ncct, return_features=True)
            aux_outputs = []

        infarct_pred = segmentation_pred[:, 0:1]
        ischemic_pred = segmentation_pred[:, 1:2]
        attention_map = self.attention_module(segmentation_pred, brain_mask)

        if self.use_ratio_constraint:
            ratio_pred = self.ratio_classifier(features)
            ratio_loss = F.binary_cross_entropy(ratio_pred, ratio_labels)
            ratio_loss = ratio_loss + 0.5 * ratio_consistency_loss(
                infarct_pred, ischemic_pred, ratio_labels, self.ratio_threshold
            )

        infarct_loss = combined_seg_loss(infarct_pred, infarct)
        ischemic_loss = combined_seg_loss(ischemic_pred, ischemic)
        aux_loss = torch.tensor(0.0, device=self.device)
        if aux_outputs:
            for aux_out in aux_outputs:
                aux_loss = aux_loss + 0.5 * (
                    combined_seg_loss(aux_out[:, 0:1], infarct) + combined_seg_loss(aux_out[:, 1:2], ischemic)
                )
            aux_loss = aux_loss / len(aux_outputs)
        segmentation_loss = infarct_loss + ischemic_loss + aux_loss

        diffusion_loss = torch.tensor(0.0, device=self.device)
        if "tmax" in batch:
            tmax = batch["tmax"].to(self.device)
            timestep = torch.randint(min_t, max_t, (ncct.shape[0],), device=self.device).long()
            diffusion_loss = self.diffusion.p_losses(
                x_start=tmax,
                ncct=ncct,
                attention_map=attention_map,
                text_features=None,
                t=timestep,
                brain_mask=brain_mask,
                use_text_attention=False,
                seg_strength=seg_strength,
            )

        base_loss = self.adaptive_loss_weights(segmentation_loss, diffusion_loss, epoch, total_epochs)
        total_loss = base_loss + self.ratio_loss_weight * ratio_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.segmentation_model.parameters())
            + list(self.attention_module.parameters())
            + list(self.diffusion_model.parameters())
            + ([] if self.ratio_classifier is None else list(self.ratio_classifier.parameters())),
            max_norm=1.0,
        )
        self.optimizer.step()
        self.scheduler.step()

        return {
            "total_loss": float(total_loss.detach().cpu()),
            "segmentation_loss": float(segmentation_loss.detach().cpu()),
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "ratio_loss": float(ratio_loss.detach().cpu()),
            "infarct_loss": float(infarct_loss.detach().cpu()),
            "ischemic_loss": float(ischemic_loss.detach().cpu()),
            "aux_loss": float(aux_loss.detach().cpu()),
            "seg_strength": float(seg_strength),
        }

    def train(self, num_epochs=1, save_every=2000):
        if self.train_loader is None:
            raise RuntimeError("train_loader is not initialized. Set build_loaders=True for real training.")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.segmentation_model.train()
            self.attention_module.train()
            self.diffusion_model.train()
            if self.ratio_classifier is not None:
                self.ratio_classifier.train()

            losses_for_epoch = []
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                losses = self.train_step(batch, epoch, num_epochs)
                losses_for_epoch.append(losses)
                for key, value in losses.items():
                    self.writer.add_scalar(f"train/{key}", value, self.step)
                if self.step > 0 and self.step % save_every == 0:
                    self.save_checkpoint(self.step)
                self.step += 1

            avg = {key: np.mean([item[key] for item in losses_for_epoch]) for key in losses_for_epoch[0]}
            print(f"Epoch {epoch}: " + ", ".join(f"{key}={value:.6f}" for key, value in avg.items()))

    def save_checkpoint(self, step):
        checkpoint = {
            "step": step,
            "current_epoch": self.current_epoch,
            "segmentation_model": self.segmentation_model.state_dict(),
            "attention_module": self.attention_module.state_dict(),
            "diffusion_model": self.diffusion_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "image_size": self.image_size,
        }
        if self.ratio_classifier is not None:
            checkpoint["ratio_classifier"] = self.ratio_classifier.state_dict()
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=" ")
    parser.add_argument("--dinov3_model_path", default=default_dinov3_path())
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--disable_progressive_training", action="store_true")
    parser.add_argument("--disable_adaptive_loss_weights", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = MultimodalDINOTrainer(
        data_root=args.data_root,
        dinov3_model_path=args.dinov3_model_path,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_lr=args.backbone_lr,
        image_size=args.image_size,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        warmup_epochs=args.warmup_epochs,
        use_progressive_training=not args.disable_progressive_training,
        use_adaptive_loss_weights=not args.disable_adaptive_loss_weights,
        build_loaders=True,
    )
    trainer.train(num_epochs=args.epochs)
