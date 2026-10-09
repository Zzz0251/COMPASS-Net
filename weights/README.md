# DINOv3 weights

Place the Hugging Face-format DINOv3 ViT-B/16 checkpoint directory here, for example:

```text
weights/
└── dinov3-vitb16-pretrain-lvd1689m/
    ├── config.json
    ├── preprocessor_config.json
    └── model.safetensors
```

Pass the directory to `--dinov3-model-path`. Model weights are intentionally excluded from this repository. Keep the checkpoint's upstream license alongside the downloaded weights.

