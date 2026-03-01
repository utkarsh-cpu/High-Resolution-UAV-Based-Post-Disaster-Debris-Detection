"""
Hurricane Debris Detection — Main Entry Point
==============================================
Replaces the monolithic first_draft.py with a modular CLI.

Usage:
    python main.py --full-pipeline
    python main.py --train-florence
    python main.py --train-sam2
    python main.py --evaluate
    python main.py --infer --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from hurricane_debris.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    Florence2Config,
    SAM2Config,
)
from hurricane_debris.utils.logging import setup_logger, get_logger

logger = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hurricane Debris Detection — Training & Inference"
    )

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full-pipeline", action="store_true", help="Run full training pipeline")
    mode.add_argument("--train-florence", action="store_true", help="Train Florence-2 only")
    mode.add_argument("--train-sam2", action="store_true", help="Train SAM2 only")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate models")
    mode.add_argument("--infer", action="store_true", help="Run cascaded inference")

    # Paths
    parser.add_argument("--dataset-dir", default="./datasets/rescuenet", help="Dataset root")
    parser.add_argument("--dataset", default="rescuenet", choices=["rescuenet", "msnet", "designsafe", "coco"])
    parser.add_argument("--florence-dir", default="./models/florence2_debris")
    parser.add_argument("--sam2-dir", default="./models/sam2_debris")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--image", type=str, help="Image path for inference")
    parser.add_argument("--output-json", type=str, help="Output JSON path for inference results")

    # Hyperparameters
    parser.add_argument("--epochs-florence", type=int, default=10)
    parser.add_argument("--epochs-sam2", type=int, default=20)
    parser.add_argument("--lr-florence", type=float, default=5e-5)
    parser.add_argument("--lr-sam2", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-file", default="./logs/training.log")

    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    """Build ExperimentConfig from CLI arguments."""
    return ExperimentConfig(
        data=DataConfig(
            dataset_root=args.dataset_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
        ),
        florence2=Florence2Config(
            num_epochs=args.epochs_florence,
            learning_rate=args.lr_florence,
            output_dir=args.florence_dir,
        ),
        sam2=SAM2Config(
            checkpoint_path=args.sam2_checkpoint,
            num_epochs=args.epochs_sam2,
            learning_rate=args.lr_sam2,
            output_dir=args.sam2_dir,
        ),
        evaluation=EvalConfig(),
        device=args.device,
        log_file=args.log_file,
    )


def load_dataset(args, config: ExperimentConfig, split: str):
    """Load the appropriate dataset based on CLI args."""
    if args.dataset == "rescuenet":
        from hurricane_debris.data.rescuenet import RescueNetDataset
        return RescueNetDataset(
            root_dir=args.dataset_dir,
            split=split,
            config=config.data,
            task="combined",
        )
    elif args.dataset == "msnet":
        from hurricane_debris.data.msnet import MSNetDataset
        return MSNetDataset(
            root_dir=args.dataset_dir,
            split=split,
            config=config.data,
            task="combined",
        )
    elif args.dataset == "designsafe":
        from hurricane_debris.data.designsafe import DesignSafeDataset
        return DesignSafeDataset(
            root_dir=args.dataset_dir,
            split=split,
            config=config.data,
        )
    else:
        from hurricane_debris.data.base_dataset import DebrisDataset
        return DebrisDataset(
            root_dir=args.dataset_dir,
            split=split,
            config=config.data,
            task="combined",
        )


def train_florence(args, config: ExperimentConfig):
    from hurricane_debris.models.florence2 import Florence2Trainer

    logger.info("=" * 60)
    logger.info("TRAINING FLORENCE-2")
    logger.info("=" * 60)

    train_ds = load_dataset(args, config, "train")
    val_ds = load_dataset(args, config, "val")

    trainer = Florence2Trainer(config=config.florence2, device=config.resolve_device())
    trainer.setup_lora()
    trainer.train(train_ds, val_ds)


def train_sam2(args, config: ExperimentConfig):
    from hurricane_debris.models.sam2_trainer import SAM2Trainer

    logger.info("=" * 60)
    logger.info("TRAINING SAM2")
    logger.info("=" * 60)

    train_ds = load_dataset(args, config, "train")
    val_ds = load_dataset(args, config, "val")

    trainer = SAM2Trainer(config=config.sam2, device=config.resolve_device())
    trainer.setup_fine_tuning()
    trainer.train(train_ds, val_ds)


def evaluate(args, config: ExperimentConfig):
    from hurricane_debris.evaluation.metrics import Evaluator

    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    evaluator = Evaluator(config=config.evaluation)
    test_ds = load_dataset(args, config, "test")

    # TODO: Run model inference on test set and accumulate metrics
    # This requires loading the trained models and running the cascade
    logger.info("Evaluation on %d test samples", len(test_ds))
    logger.info(evaluator.summary())


def infer(args, config: ExperimentConfig):
    from hurricane_debris.models.cascade import CascadedInference

    logger.info("=" * 60)
    logger.info("CASCADED INFERENCE")
    logger.info("=" * 60)

    if not args.image:
        logger.error("--image is required for inference mode")
        sys.exit(1)

    pipeline = CascadedInference(
        florence_model_dir=args.florence_dir,
        sam2_checkpoint=args.sam2_dir + "/best_model.pth",
        config=config,
        device=config.resolve_device(),
    )

    result = pipeline.run(args.image)
    print(json.dumps(result.to_json(), indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result.to_json(), f, indent=2)
        logger.info("Results saved to %s", args.output_json)


def main():
    args = parse_args()
    config = build_config(args)

    setup_logger(log_file=config.log_file)
    logger.info("Device: %s", config.resolve_device())

    if args.full_pipeline:
        train_florence(args, config)
        train_sam2(args, config)
        evaluate(args, config)
    elif args.train_florence:
        train_florence(args, config)
    elif args.train_sam2:
        train_sam2(args, config)
    elif args.evaluate:
        evaluate(args, config)
    elif args.infer:
        infer(args, config)
    else:
        print("""
Hurricane Debris Detection — Training & Inference
==================================================

Usage:
  python main.py --full-pipeline                     # Full training + evaluation
  python main.py --train-florence --dataset rescuenet # Train Florence-2
  python main.py --train-sam2 --dataset rescuenet     # Train SAM2
  python main.py --evaluate --dataset msnet           # Cross-dataset evaluation
  python main.py --infer --image img.jpg              # Cascaded inference

Options:
  --dataset-dir PATH       Dataset root (default: ./datasets/rescuenet)
  --dataset NAME           rescuenet | msnet | designsafe | coco
  --epochs-florence N      Florence-2 epochs (default: 10)
  --epochs-sam2 N          SAM2 epochs (default: 20)
  --image-size N           Input resolution (default: 768)
  --device DEVICE          auto | cuda | cpu | mps
        """)


if __name__ == "__main__":
    main()
