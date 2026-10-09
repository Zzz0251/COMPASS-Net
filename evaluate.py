from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from compass.backbone import CoMPASSNet
from compass.data import create_dataloader
from compass.diffusion import DiffusionUNet, GaussianDiffusion
from compass.losses import derive_penumbra
from compass.trainer import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-level 3D evaluation for CoMPASS-Net")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dinov3-model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="predictions/test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-diffusion", action="store_true")
    return parser.parse_args()


def dice_3d(prediction: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    return float((2.0 * intersection + smooth) / (prediction.sum() + target.sum() + smooth))


def save_binary(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


def save_continuous(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().float().cpu().numpy()
    minimum, maximum = float(array.min()), float(array.max())
    if maximum > minimum:
        array = (array - minimum) / (maximum - minimum)
    else:
        array = np.zeros_like(array)
    Image.fromarray((array * 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    directories = {
        name: output_dir / name
        for name in ("infarct", "ischemic", "penumbra", "attention")
    }
    if args.save_diffusion:
        directories["pseudo_tmax"] = output_dir / "pseudo_tmax"
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint.get("config", {}))
    loader = create_dataloader(
        args.data_root,
        split=args.split,
        batch_size=1,
        image_size=config.image_size,
        num_workers=args.num_workers,
        require_tmax=False,
    )
    model = CoMPASSNet(
        args.dinov3_model_path,
        image_size=config.image_size,
        local_files_only=not args.allow_download,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    diffusion = None
    if args.save_diffusion:
        denoiser = DiffusionUNet(base_dim=64, self_condition=True).to(device)
        denoiser.load_state_dict(checkpoint["denoiser"])
        denoiser.eval()
        diffusion = GaussianDiffusion(
            denoiser,
            image_size=config.image_size,
            timesteps=config.diffusion_timesteps,
            sampling_timesteps=config.sampling_timesteps,
        ).to(device)

    volumes: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    slice_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluation"):
            ncct = batch["ncct"].to(device)
            brain_mask = batch["brain_mask"].to(device)
            output = model(ncct, brain_mask)
            logits = output["logits"]
            attention = output["attention_map"]
            ratio_probability = output["ratio_probability"]
            assert isinstance(logits, torch.Tensor)
            assert isinstance(attention, torch.Tensor)
            assert isinstance(ratio_probability, torch.Tensor)

            prediction = torch.sigmoid(logits) >= args.threshold
            infarct = prediction[0, 0]
            ischemic = prediction[0, 1]
            penumbra = derive_penumbra(ischemic, infarct)
            name = batch["filename"][0]
            patient = batch["patient_id"][0]
            save_binary(directories["infarct"] / f"{name}.png", infarct)
            save_binary(directories["ischemic"] / f"{name}.png", ischemic)
            save_binary(directories["penumbra"] / f"{name}.png", penumbra)
            save_continuous(directories["attention"] / f"{name}.png", attention[0, 0])

            gt_infarct = batch["infarct"][0, 0].numpy() >= 0.5
            gt_ischemic = batch["ischemic"][0, 0].numpy() >= 0.5
            gt_penumbra = np.logical_and(gt_ischemic, np.logical_not(gt_infarct))
            for key, value in (
                ("pred_infarct", infarct.cpu().numpy()),
                ("pred_ischemic", ischemic.cpu().numpy()),
                ("pred_penumbra", penumbra.cpu().numpy()),
                ("gt_infarct", gt_infarct),
                ("gt_ischemic", gt_ischemic),
                ("gt_penumbra", gt_penumbra),
            ):
                volumes[patient][key].append(np.asarray(value, dtype=bool))

            record = {
                "filename": name,
                "patient_id": patient,
                "mismatch_probability": float(ratio_probability.item()),
            }
            if diffusion is not None:
                pseudo_tmax = diffusion.sample(attention, brain_mask, show_progress=False)
                save_continuous(directories["pseudo_tmax"] / f"{name}.png", pseudo_tmax[0, 0])
            slice_records.append(record)

    patient_metrics = []
    for patient, arrays in sorted(volumes.items()):
        stacked = {key: np.stack(value, axis=0) for key, value in arrays.items()}
        patient_metrics.append(
            {
                "patient_id": patient,
                "infarct_dice": dice_3d(stacked["pred_infarct"], stacked["gt_infarct"]),
                "ischemic_dice": dice_3d(stacked["pred_ischemic"], stacked["gt_ischemic"]),
                "penumbra_dice": dice_3d(stacked["pred_penumbra"], stacked["gt_penumbra"]),
                "slices": int(stacked["pred_infarct"].shape[0]),
            }
        )
    mean_metrics = {
        key: float(np.mean([patient[key] for patient in patient_metrics]))
        for key in ("infarct_dice", "ischemic_dice", "penumbra_dice")
    }
    result = {"mean_3d_dice": mean_metrics, "patients": patient_metrics, "slices": slice_records}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["mean_3d_dice"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

