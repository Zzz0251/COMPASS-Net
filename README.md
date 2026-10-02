# DINOv3 

The segmentation backbone is DINOv3, with the diffusion and mismatch-ratio branches.

## How to train

```bash
python main.py --data_root /data/xx --epochs 200 --batch_size 4
```

data for train:
    -brain_mask (for diffusion branch constraint in .npy format)
    -train
        -ncct or images (2d png / npz / pt)
        -infarct
        -ischemic tissue (the old folder/key name `penumbra` is also accepted)
        -Tmax (optional, for diffusion supervision)
    -test
        -same structure as train


The default slice size is `224x224`, which matches DINOv3 input.

## Files and folders

- `main.py`: main training entry. It keeps the original training strategy and replaces the segmentation backbone with DINOv3.
- `dino_segmentation_model.py`: DINOv3 segmentation backbone and multi-scale ResUNet-style decoder with skip connections.
- `models/`: diffusion and segmentation branches
- `data/dataset.py`: creates dataloaders and resizes slices/masks to `224x224`.
- `dino_reference/weights/dinov3-vitb16-pretrain-lvd1689m/`: default DINOv3 model path loaded by the training script.

The two segmentation channels are infarct core and ischemic tissue. The mismatch ratio is computed as `area(ischemic tissue) / area(infarct core)`.

TODO : This is the code version in my early experiment stage, and I am still working on the final version. This version also done under the help of codex. Thanks for the understanding. 
