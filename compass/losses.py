from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.where(target == 1, probability, 1.0 - probability)
    return (alpha * (1.0 - pt).pow(gamma) * cross_entropy).mean()


def combined_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.7 * dice_loss(logits, target) + 0.3 * focal_loss(logits, target)


def mismatch_ratio(
    infarct: torch.Tensor,
    ischemic: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """ResNet-compatible salvage/core mismatch: (ischemic - core) / core."""
    infarct_area = infarct.sum(dim=(1, 2, 3))
    ischemic_area = ischemic.sum(dim=(1, 2, 3))
    return (ischemic_area - infarct_area + smooth) / (infarct_area + smooth)


def mismatch_labels(
    infarct: torch.Tensor,
    ischemic: torch.Tensor,
    threshold: float = 1.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = mismatch_ratio(infarct, ischemic)
    return (ratio >= threshold).float(), ratio


def mismatch_consistency_loss(
    infarct_logits: torch.Tensor,
    ischemic_logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 1.8,
) -> torch.Tensor:
    predicted_ratio = mismatch_ratio(torch.sigmoid(infarct_logits), torch.sigmoid(ischemic_logits))
    classification_logit = (predicted_ratio - threshold) * 5.0
    return F.binary_cross_entropy_with_logits(classification_logit, labels)


def segmentation_loss(
    logits: torch.Tensor,
    aux_logits: list[torch.Tensor],
    infarct: torch.Tensor,
    ischemic: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    infarct_loss = combined_segmentation_loss(logits[:, 0:1], infarct)
    ischemic_loss = combined_segmentation_loss(logits[:, 1:2], ischemic)
    auxiliary = logits.new_zeros(())
    if aux_logits:
        auxiliary = torch.stack(
            [
                0.5
                * (
                    combined_segmentation_loss(aux[:, 0:1], infarct)
                    + combined_segmentation_loss(aux[:, 1:2], ischemic)
                )
                for aux in aux_logits
            ]
        ).mean()
    total = infarct_loss + ischemic_loss + auxiliary
    return total, {"infarct": infarct_loss, "ischemic": ischemic_loss, "auxiliary": auxiliary}


def binary_dice(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = (prediction * target).sum()
    return (2.0 * intersection + smooth) / (prediction.sum() + target.sum() + smooth)


def derive_penumbra(ischemic: torch.Tensor, infarct: torch.Tensor) -> torch.Tensor:
    return torch.logical_and(ischemic.bool(), torch.logical_not(infarct.bool()))
