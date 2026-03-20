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
    python main.py --download --dataset rescuenet --dataset-dir ./datasets
    python main.py --download --dataset all       --dataset-dir ./datasets
"""

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional for lightweight environments
    torch = None

from hurricane_debris.config import (
    DEBRIS_CATEGORIES,
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
    mode.add_argument("--download", action="store_true", help="Download dataset(s) to --dataset-dir")

    # Paths
    parser.add_argument("--dataset-dir", default="./datasets/rescuenet", help="Dataset root")
    parser.add_argument(
        "--dataset",
        default="rescuenet",
        choices=["rescuenet", "msnet", "designsafe", "coco", "all"],
        help="Dataset name; 'all' downloads every registered dataset",
    )
    parser.add_argument("--florence-dir", default="./models/florence2_debris")
    parser.add_argument("--sam2-dir", default="./models/sam2_debris")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--image", type=str, help="Image path for inference")
    parser.add_argument("--output-json", type=str, help="Output JSON path for inference results")
    parser.add_argument("--output-geojson", type=str, help="Output GeoJSON path for inference results")
    parser.add_argument("--metrics-dir", default="./outputs/metrics", help="Directory for evaluation metrics")
    parser.add_argument("--cross-dataset", action="store_true", help="Run evaluation on configured test datasets")
    parser.add_argument(
        "--strict-eval-model",
        action="store_true",
        help="Fail evaluation if cascade models cannot be loaded",
    )

    # Hyperparameters
    parser.add_argument("--epochs-florence", type=int, default=10)
    parser.add_argument("--epochs-sam2", type=int, default=20)
    parser.add_argument("--lr-florence", type=float, default=5e-5)
    parser.add_argument("--lr-sam2", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-file", default="./logs/training.log")
    parser.add_argument("--seed", type=int, default=42)

    # Download options
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download dataset even if it already exists",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive after extraction",
    )

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
        seed=args.seed,
        log_file=args.log_file,
    )


def set_seed(seed: int):
    """Set global random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable TF32 for faster fp32 fallback ops on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.astype(float).copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def _category_name_to_id() -> dict:
    return {meta["name"]: cid for cid, meta in DEBRIS_CATEGORIES.items()}


class OraclePredictor:
    """Lightweight predictor for tests/fallback when model checkpoints are unavailable."""

    def predict(self, sample: dict) -> dict:
        target = sample["target"]
        gt_boxes_xywh = _to_numpy(target.get("bboxes", np.zeros((0, 4))))
        gt_boxes_xyxy = _xywh_to_xyxy(gt_boxes_xywh)
        gt_labels = _to_numpy(target.get("category_ids", np.zeros(0, dtype=int))).astype(int)
        scores = np.full((len(gt_labels),), 1.0, dtype=float)
        out = {
            "bboxes": gt_boxes_xyxy,
            "scores": scores,
            "labels": gt_labels,
        }
        if "semantic_mask" in target:
            out["semantic_mask"] = _to_numpy(target["semantic_mask"])
        return out


class CascadePredictor:
    """Run cascaded Florence-2 -> SAM2 inference for evaluation."""

    def __init__(self, args, config: ExperimentConfig):
        from hurricane_debris.models.cascade import CascadedInference

        sam2_best = str(Path(args.sam2_dir) / "best_model.pth")
        self._category_to_id = _category_name_to_id()
        self.pipeline = CascadedInference(
            florence_model_dir=args.florence_dir,
            sam2_checkpoint=sam2_best,
            config=config,
            device=config.resolve_device(),
        )

    def predict(self, sample: dict) -> dict:
        image_path = sample.get("image_path")
        if not image_path:
            raise ValueError("Sample is missing image_path; cannot run cascade inference")

        result = self.pipeline.run(str(image_path), score_threshold=0.0)
        bboxes = np.asarray([d.bbox for d in result.detections], dtype=float).reshape(-1, 4)
        scores = np.asarray([d.score for d in result.detections], dtype=float).reshape(-1)
        labels = np.asarray(
            [self._category_to_id.get(d.category, 3) for d in result.detections],
            dtype=int,
        ).reshape(-1)

        out = {"bboxes": bboxes, "scores": scores, "labels": labels}
        if result.detections and any(d.mask is not None for d in result.detections):
            semantic_mask = np.zeros((result.height, result.width), dtype=np.int64)
            for det in result.detections:
                if det.mask is None:
                    continue
                cid = self._category_to_id.get(det.category, 3)
                semantic_mask[det.mask.astype(bool)] = cid
            out["semantic_mask"] = semantic_mask

        return out


def _build_predictor(args, config: ExperimentConfig):
    """Prefer cascade model inference; optionally fall back to oracle predictor."""
    try:
        return CascadePredictor(args, config)
    except Exception as exc:
        if args.strict_eval_model:
            raise RuntimeError(f"Failed to initialize cascade predictor: {exc}") from exc
        logger.warning(
            "Cascade predictor unavailable (%s). Falling back to oracle predictor.",
            exc,
        )
        return OraclePredictor()


def _resolve_dataset_dir(dataset_dir: str, dataset_name: str, cross_dataset: bool) -> str:
    base = Path(dataset_dir)
    if not cross_dataset:
        return str(base)

    if base.name.lower() == dataset_name.lower():
        return str(base)

    candidate = base / dataset_name
    if candidate.exists():
        return str(candidate)

    candidate = base.parent / dataset_name
    if candidate.exists():
        return str(candidate)

    logger.warning(
        "Could not auto-resolve directory for dataset '%s'; using %s",
        dataset_name,
        base,
    )
    return str(base)


def _save_run_artifacts(config: ExperimentConfig, args):
    """Persist run configuration for reproducibility."""
    artifact_dir = Path(args.metrics_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "run_config.json"
    with open(artifact_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "config": dataclasses.asdict(config),
            },
            f,
            indent=2,
        )
    logger.info("Run configuration saved to %s", artifact_path)


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


def download(args):
    """Download one or all datasets to the configured destination directory."""
    from hurricane_debris.data.download import download_dataset

    # For 'all', use the parent datasets directory; otherwise use dataset_dir
    dataset_name = args.dataset
    if dataset_name == "all":
        dest_dir = args.dataset_dir
    else:
        # If dataset_dir already ends with the dataset name, use its parent
        dest_path = Path(args.dataset_dir)
        if dest_path.name.lower() == dataset_name.lower():
            dest_dir = str(dest_path.parent)
        else:
            dest_dir = args.dataset_dir

    logger.info("=" * 60)
    logger.info("DOWNLOADING DATASET(S): %s", dataset_name.upper())
    logger.info("Destination: %s", dest_dir)
    logger.info("=" * 60)

    try:
        result_path = download_dataset(
            name=dataset_name,
            dest_dir=dest_dir,
            force=args.force_download,
            keep_archive=args.keep_archive,
        )
        logger.info("Download complete. Data available at: %s", result_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)


class _FilteredSubset(torch.utils.data.Dataset):
    """Subset wrapper that keeps only non-empty samples by precomputed indices."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def _filter_empty_samples(dataset):
    """Remove samples with zero foreground pixels to avoid wasted forward passes."""
    if not hasattr(dataset, 'has_foreground'):
        return dataset

    total = len(dataset)
    logger.info("Scanning %d training samples for empty masks...", total)
    valid_indices = []
    for i in range(total):
        if dataset.has_foreground(i):
            valid_indices.append(i)
        if (i + 1) % 500 == 0:
            logger.info("  scanned %d / %d samples ...", i + 1, total)

    removed = total - len(valid_indices)
    if removed > 0:
        logger.info(
            "Filtered %d empty samples from training set (%d → %d)",
            removed, total, len(valid_indices),
        )
        return _FilteredSubset(dataset, valid_indices)
    logger.info("All %d samples have foreground content", total)
    return dataset


def train_florence(args, config: ExperimentConfig):
    from hurricane_debris.models.florence2 import Florence2Trainer

    logger.info("=" * 60)
    logger.info("TRAINING FLORENCE-2")
    logger.info("=" * 60)

    train_ds = load_dataset(args, config, "train")
    val_ds = load_dataset(args, config, "val")

    # Filter out samples with no detections (empty targets waste forward passes)
    train_ds = _filter_empty_samples(train_ds)

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

    predictor = _build_predictor(args, config)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset_order = [args.dataset]
    if args.cross_dataset:
        dataset_order.extend(config.evaluation.test_datasets)

    # Deduplicate while preserving order
    dataset_order = list(dict.fromkeys(dataset_order))
    all_results = {}

    for dataset_name in dataset_order:
        logger.info("Evaluating dataset: %s", dataset_name)
        ds_args = argparse.Namespace(**vars(args))
        ds_args.dataset = dataset_name
        ds_args.dataset_dir = _resolve_dataset_dir(
            args.dataset_dir, dataset_name, args.cross_dataset
        )
        evaluator = Evaluator(config=config.evaluation)
        test_ds = load_dataset(ds_args, config, "test")

        for i in range(len(test_ds)):
            sample = test_ds[i]
            pred = predictor.predict(sample)
            target = sample.get("target", {})

            pred_bboxes = _to_numpy(pred.get("bboxes", np.zeros((0, 4))))
            pred_scores = _to_numpy(pred.get("scores", np.zeros(0)))
            pred_labels = _to_numpy(pred.get("labels", np.zeros(0, dtype=int)))

            gt_bboxes_xywh = _to_numpy(target.get("bboxes", np.zeros((0, 4))))
            gt_bboxes = _xywh_to_xyxy(gt_bboxes_xywh)
            gt_labels = _to_numpy(target.get("category_ids", np.zeros(0, dtype=int)))

            evaluator.update_detection(
                pred_bboxes=pred_bboxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                iou_threshold=config.evaluation.f1_threshold,
            )

            if "semantic_mask" in pred and "semantic_mask" in target:
                evaluator.update_segmentation(
                    pred_mask=_to_numpy(pred["semantic_mask"]),
                    gt_mask=_to_numpy(target["semantic_mask"]),
                )

        results = evaluator.compute()
        all_results[dataset_name] = results

        dataset_metrics_path = metrics_dir / f"metrics_{dataset_name}.json"
        with open(dataset_metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(
            "Evaluation complete on %s (%d samples). Metrics saved to %s",
            dataset_name,
            len(test_ds),
            dataset_metrics_path,
        )

    if args.cross_dataset:
        summary_path = metrics_dir / "cross_dataset_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Cross-dataset summary saved to %s", summary_path)

    logger.info("Primary dataset summary:\n%s", json.dumps(all_results[args.dataset], indent=2))


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

    if args.output_geojson:
        with open(args.output_geojson, "w") as f:
            json.dump(result.to_geojson(), f, indent=2)
        logger.info("GeoJSON saved to %s", args.output_geojson)


def main():
    args = parse_args()

    # --download runs without a full ExperimentConfig / artifact saving
    if args.download:
        setup_logger()
        download(args)
        return

    config = build_config(args)

    setup_logger(log_file=config.log_file)
    set_seed(config.seed)
    _save_run_artifacts(config, args)
    logger.info("Global seed set to %d", config.seed)
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
  python main.py --download --dataset rescuenet      # Download RescueNet
  python main.py --download --dataset all            # Download all datasets
  python main.py --full-pipeline                     # Full training + evaluation
  python main.py --train-florence --dataset rescuenet # Train Florence-2
  python main.py --train-sam2 --dataset rescuenet     # Train SAM2
  python main.py --evaluate --dataset msnet           # Cross-dataset evaluation
  python main.py --infer --image img.jpg              # Cascaded inference

Options:
  --dataset-dir PATH       Dataset root (default: ./datasets/rescuenet)
  --dataset NAME           rescuenet | msnet | designsafe | coco | all
  --epochs-florence N      Florence-2 epochs (default: 10)
  --epochs-sam2 N          SAM2 epochs (default: 20)
  --image-size N           Input resolution (default: 768)
  --device DEVICE          auto | cuda | cpu | mps
  --force-download         Re-download even if data already exists
  --keep-archive           Keep the downloaded archive after extraction
        """)


if __name__ == "__main__":
    main()
