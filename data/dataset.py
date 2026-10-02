import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _to_tensor(array):
    if torch.is_tensor(array):
        tensor = array.float()
    else:
        tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    return tensor


def _resize(tensor, image_size, is_mask=False):
    if tensor.shape[-2:] == (image_size, image_size):
        return tensor
    mode = "nearest" if is_mask else "bilinear"
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return tensor.squeeze(0)


def _load_image(path, image_size, is_mask=False):
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    tensor = _to_tensor(array)
    if is_mask:
        tensor = (tensor > 0.5).float()
    return _resize(tensor, image_size, is_mask=is_mask)


class StrokeSliceDataset(Dataset):

    def __init__(self, data_root, image_size=224, mode="train"):
        self.root = Path(data_root)
        self.mode = mode
        self.image_size = image_size
        self.mode_root = self.root / mode if (self.root / mode).exists() else self.root

        self.file_samples = sorted(
            [
                path
                for path in self.mode_root.iterdir()
                if path.is_file() and path.suffix.lower() in {".npz", ".pt", ".pth"}
            ]
        )

        ncct_dir = self.mode_root / "ncct"
        self.folder_samples = []
        if ncct_dir.exists():
            for path in sorted(ncct_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTS:
                    self.folder_samples.append(path)

        if not self.file_samples and not self.folder_samples:
            raise FileNotFoundError(
                f"No supported samples found in {self.mode_root}. "
            )

    def __len__(self):
        return len(self.file_samples) + len(self.folder_samples)

    def __getitem__(self, index):
        if index < len(self.file_samples):
            return self._load_file_sample(self.file_samples[index])
        return self._load_folder_sample(self.folder_samples[index - len(self.file_samples)])

    def _load_file_sample(self, path):
        if path.suffix.lower() == ".npz":
            data = dict(np.load(path))
        else:
            data = torch.load(path, map_location="cpu")

        ischemic_key = "ischemic" if "ischemic" in data else "penumbra"
        ischemic = _resize((_to_tensor(data[ischemic_key]) > 0.5).float(), self.image_size, is_mask=True)
        sample = {
            "ncct": _resize(_to_tensor(data["ncct"]), self.image_size),
            "infarct": _resize((_to_tensor(data["infarct"]) > 0.5).float(), self.image_size, is_mask=True),
            "ischemic": ischemic,
            "penumbra": ischemic,
            "filename": path.stem,
        }
        if "brain_mask" in data:
            sample["brain_mask"] = _resize((_to_tensor(data["brain_mask"]) > 0.5).float(), self.image_size, is_mask=True)
        else:
            sample["brain_mask"] = torch.ones_like(sample["infarct"])
        if "tmax" in data:
            sample["tmax"] = _resize(_to_tensor(data["tmax"]), self.image_size)
        return sample

    def _load_folder_sample(self, ncct_path):
        name = ncct_path.name
        ischemic_dir = self.mode_root / "ischemic"
        if not ischemic_dir.exists():
            ischemic_dir = self.mode_root / "penumbra"

        ischemic = _load_image(ischemic_dir / name, self.image_size, is_mask=True)
        sample = {
            "ncct": _load_image(ncct_path, self.image_size),
            "infarct": _load_image(self.mode_root / "infarct" / name, self.image_size, is_mask=True),
            "ischemic": ischemic,
            "penumbra": ischemic,
            "filename": ncct_path.stem,
        }

        brain_mask_path = self.mode_root / "brain_mask" / name
        sample["brain_mask"] = (
            _load_image(brain_mask_path, self.image_size, is_mask=True)
            if brain_mask_path.exists()
            else torch.ones_like(sample["infarct"])
        )

        tmax_path = self.mode_root / "tmax" / name
        if tmax_path.exists():
            sample["tmax"] = _load_image(tmax_path, self.image_size)
        return sample


def create_dataloader(data_root, batch_size=2, image_size=224, mode="train", num_workers=0):
    dataset = StrokeSliceDataset(data_root=data_root, image_size=image_size, mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
