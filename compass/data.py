from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
def patient_id_from_name(name: str) -> str:
    """Return the patient prefix used to rebuild a 3D volume from slice names."""
    return name.split("_slice_", 1)[0]


def _resize(tensor: torch.Tensor, image_size: int, is_mask: bool) -> torch.Tensor:
    if tensor.shape[-2:] == (image_size, image_size):
        return tensor
    mode = "nearest" if is_mask else "bilinear"
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    ).squeeze(0)


def _as_chw(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = value.detach().float() if torch.is_tensor(value) else torch.as_tensor(np.asarray(value)).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 2D image or CHW/HWC tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _scale_image(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.numel() and tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _load_gray(path: Path, image_size: int, is_mask: bool = False) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        tensor = _as_chw(np.load(path))
    else:
        tensor = _as_chw(np.asarray(Image.open(path).convert("L"), dtype=np.float32))
    tensor = _scale_image(tensor)
    if is_mask:
        tensor = (tensor > 0.5).float()
    return _resize(tensor, image_size, is_mask=is_mask)


def _normalize_tmax(tmax: torch.Tensor) -> torch.Tensor:
    """Preserve the original ResNet pipeline's per-slice z-score normalization."""
    std = tmax.std(unbiased=False)
    return (tmax - tmax.mean()) / std.clamp_min(1e-8)


def _find_by_stem(directory: Path, stem: str) -> Optional[Path]:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


class StrokeSliceDataset(Dataset):
    """Load the original composite layout or an explicit-folder/NPZ layout."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int = 224,
        require_tmax: bool = False,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.require_tmax = require_tmax
        self.split_root = self.root / split
        if not self.split_root.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_root}")

        if (self.split_root / "images").is_dir():
            self.layout = "composite"
            self.samples = self._image_files(self.split_root / "images")
        elif (self.split_root / "ncct").is_dir():
            self.layout = "folders"
            self.samples = self._image_files(self.split_root / "ncct")
        else:
            self.layout = "npz"
            self.samples = sorted(self.split_root.glob("*.npz"))

        if not self.samples:
            raise FileNotFoundError(f"No supported samples found under {self.split_root}")
        self._validate_samples()

    @staticmethod
    def _image_files(directory: Path) -> list[Path]:
        return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    def _validate_samples(self) -> None:
        errors = []
        for path in self.samples:
            stem = path.stem
            if self.layout == "composite":
                brain = self.root / "bmsk" / f"{stem}.npy"
                if not brain.exists():
                    errors.append(f"missing brain mask: {brain}")
                if self.require_tmax and _find_by_stem(self.split_root / "Tmax", stem) is None:
                    errors.append(f"missing Tmax: {self.split_root / 'Tmax' / stem}")
            elif self.layout == "folders":
                ischemic_dir = self.split_root / "ischemic"
                if not ischemic_dir.exists():
                    ischemic_dir = self.split_root / "penumbra"
                for label, directory in (("infarct", self.split_root / "infarct"), ("ischemic", ischemic_dir)):
                    if _find_by_stem(directory, stem) is None:
                        errors.append(f"missing {label} mask for {path.name}")
                tmax_dir = self.split_root / "tmax"
                if not tmax_dir.exists():
                    tmax_dir = self.split_root / "Tmax"
                if self.require_tmax and _find_by_stem(tmax_dir, stem) is None:
                    errors.append(f"missing Tmax for {path.name}")
            if len(errors) >= 20:
                break
        if errors:
            details = "\n".join(f"- {item}" for item in errors)
            raise FileNotFoundError(f"Dataset integrity check failed:\n{details}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        path = self.samples[index]
        if self.layout == "composite":
            sample = self._load_composite(path)
        elif self.layout == "folders":
            sample = self._load_folders(path)
        else:
            sample = self._load_npz(path)
        sample["filename"] = path.stem
        sample["patient_id"] = patient_id_from_name(path.stem)
        return sample

    def _load_composite(self, path: Path) -> Dict[str, torch.Tensor]:
        if path.suffix.lower() == ".npy":
            image = _as_chw(np.load(path))
        else:
            image = _as_chw(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))
        image = _scale_image(image)
        if image.shape[0] < 3:
            raise ValueError(f"Composite image must have three channels: {path}")
        sample = {
            "ncct": _resize(image[0:1], self.image_size, is_mask=False),
            "infarct": _resize((image[1:2] > 0.5).float(), self.image_size, is_mask=True),
            "ischemic": _resize((image[2:3] > 0.5).float(), self.image_size, is_mask=True),
            "brain_mask": _load_gray(self.root / "bmsk" / f"{path.stem}.npy", self.image_size, is_mask=True),
        }
        tmax_path = _find_by_stem(self.split_root / "Tmax", path.stem)
        if tmax_path is not None:
            sample["tmax"] = _normalize_tmax(_load_gray(tmax_path, self.image_size))
        return sample

    def _load_folders(self, path: Path) -> Dict[str, torch.Tensor]:
        ischemic_dir = self.split_root / "ischemic"
        if not ischemic_dir.exists():
            ischemic_dir = self.split_root / "penumbra"
        infarct_path = _find_by_stem(self.split_root / "infarct", path.stem)
        ischemic_path = _find_by_stem(ischemic_dir, path.stem)
        assert infarct_path is not None and ischemic_path is not None
        sample = {
            "ncct": _load_gray(path, self.image_size),
            "infarct": _load_gray(infarct_path, self.image_size, is_mask=True),
            "ischemic": _load_gray(ischemic_path, self.image_size, is_mask=True),
        }
        brain_path = _find_by_stem(self.split_root / "brain_mask", path.stem)
        sample["brain_mask"] = (
            _load_gray(brain_path, self.image_size, is_mask=True)
            if brain_path is not None
            else torch.ones_like(sample["infarct"])
        )
        tmax_dir = self.split_root / "tmax"
        if not tmax_dir.exists():
            tmax_dir = self.split_root / "Tmax"
        tmax_path = _find_by_stem(tmax_dir, path.stem)
        if tmax_path is not None:
            sample["tmax"] = _normalize_tmax(_load_gray(tmax_path, self.image_size))
        return sample

    def _load_npz(self, path: Path) -> Dict[str, torch.Tensor]:
        with np.load(path) as data:
            ischemic_key = "ischemic" if "ischemic" in data else "penumbra"
            sample = {
                "ncct": _resize(_scale_image(_as_chw(data["ncct"])), self.image_size, False),
                "infarct": _resize((_as_chw(data["infarct"]) > 0.5).float(), self.image_size, True),
                "ischemic": _resize((_as_chw(data[ischemic_key]) > 0.5).float(), self.image_size, True),
            }
            sample["brain_mask"] = (
                _resize((_as_chw(data["brain_mask"]) > 0.5).float(), self.image_size, True)
                if "brain_mask" in data
                else torch.ones_like(sample["infarct"])
            )
            if "tmax" in data:
                sample["tmax"] = _normalize_tmax(_resize(_as_chw(data["tmax"]), self.image_size, False))
        if self.require_tmax and "tmax" not in sample:
            raise KeyError(f"Training sample has no Tmax array: {path}")
        return sample


def create_dataloader(
    data_root: str | Path,
    split: str,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 4,
    require_tmax: bool = False,
) -> DataLoader:
    dataset = StrokeSliceDataset(data_root, split=split, image_size=image_size, require_tmax=require_tmax)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train" and len(dataset) >= batch_size,
        persistent_workers=num_workers > 0,
    )
