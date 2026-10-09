# CoMPASS-Net: DINOv3 main pipeline

This directory contains the main experimental pipeline for the paper Perfusion Aware Infarct Core and Penumbra Segmentation on NCCT with Region Constraints.

The implementation uses:

-- a locally loaded DINOv3 ViT-B/16 encoder;

-- a ResUNet-style decoder with two output heads for infarct core and total ischemic tissue;

-- the mismatch-ratio classification and consistency constraint;

-- a segmentation-guided conditional diffusion auxiliary branch supervised by Tmax during training;

-- 3D patient-level Dice evaluation, with penumbra derived as ischemic AND NOT infarct.


Note: the code is prepared with the help of codex, so if you have any issue, please contact me for more precise details.
