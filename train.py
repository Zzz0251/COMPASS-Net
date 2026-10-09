from __future__ import annotations

import argparse
import json
from pathlib import Path

from compass.trainer import ExperimentConfig, MainExperimentTrainer


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the CoMPASS-Net DINOv3 main experiment")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dinov3-model-path", required=True)
    parser.add_argument("--output-dir", default="runs/main")
    parser.add_argument("--config", default=str(root / "configs" / "main_experiment.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    for argument, attribute in (
        (args.batch_size, "batch_size"),
        (args.epochs, "epochs"),
        (args.learning_rate, "learning_rate"),
        (args.num_workers, "num_workers"),
    ):
        if argument is not None:
            setattr(config, attribute, argument)

    trainer = MainExperimentTrainer(
        data_root=args.data_root,
        dinov3_model_path=args.dinov3_model_path,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        local_files_only=not args.allow_download,
        use_amp=not args.no_amp,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    if args.dry_run:
        print(json.dumps(trainer.dry_run(), indent=2, sort_keys=True))
        return
    trainer.train()


if __name__ == "__main__":
    main()

