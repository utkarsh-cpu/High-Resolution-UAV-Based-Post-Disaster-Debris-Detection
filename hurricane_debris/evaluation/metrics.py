"""
Evaluation Metrics
==================
- mIoU  — mean Intersection over Union for semantic segmentation
- F1    — detection F1-score at a given IoU threshold
- AP    — Average Precision at IoU ∈ [0.5 : 0.05 : 0.95]

Designed to work with both the cascaded pipeline outputs and raw
model predictions during training validation.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hurricane_debris.config import EvalConfig
from hurricane_debris.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


class Evaluator:
    """
    Compute mIoU, F1, and COCO-style AP for debris detection/segmentation.

    Usage::

        evaluator = Evaluator(num_classes=7)
        for batch in dataloader:
            preds, targets = model(batch), batch["target"]
            evaluator.update(preds, targets)
        results = evaluator.compute()
        evaluator.reset()
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.cfg = config or EvalConfig()
        self.num_classes = self.cfg.num_classes

        # ── Confusion matrix for mIoU ────────────────────────────────────
        # Shape: (num_classes + 1, num_classes + 1)  including background=0
        self._confusion = np.zeros(
            (self.num_classes + 1, self.num_classes + 1), dtype=np.int64
        )

        # ── Detection records for F1 / AP ────────────────────────────────
        # Each entry: (score, is_tp, category_id)
        self._det_records: List[Tuple[float, bool, int]] = []
        self._n_gt: Dict[int, int] = defaultdict(int)  # ground-truth count per class

    def reset(self):
        """Clear accumulated statistics."""
        self._confusion[:] = 0
        self._det_records.clear()
        self._n_gt.clear()

    # ── Update methods ───────────────────────────────────────────────────

    def update_segmentation(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ):
        """
        Accumulate a pair of semantic segmentation masks.

        Args:
            pred_mask: [H, W] predicted class IDs (int).
            gt_mask:   [H, W] ground-truth class IDs (int).
        """
        assert pred_mask.shape == gt_mask.shape, (
            f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}"
        )
        n = self.num_classes + 1
        mask = (gt_mask >= 0) & (gt_mask < n) & (pred_mask >= 0) & (pred_mask < n)
        self._confusion += np.bincount(
            n * gt_mask[mask].astype(int) + pred_mask[mask].astype(int),
            minlength=n * n,
        ).reshape(n, n)

    def update_detection(
        self,
        pred_bboxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_bboxes: np.ndarray,
        gt_labels: np.ndarray,
        iou_threshold: float = 0.5,
    ):
        """
        Accumulate one image's detection results.

        Bboxes in [x1, y1, x2, y2] format. Labels as integer class IDs.
        """
        # Count ground truths
        for gl in gt_labels:
            self._n_gt[int(gl)] += 1

        if len(pred_bboxes) == 0:
            return

        if len(gt_bboxes) == 0:
            for score, label in zip(pred_scores, pred_labels):
                self._det_records.append((float(score), False, int(label)))
            return

        # Compute IoU matrix [N_pred, N_gt]
        iou_matrix = self._compute_iou_matrix(pred_bboxes, gt_bboxes)

        matched_gt = set()
        # Sort predictions by descending score
        order = np.argsort(-pred_scores)

        for idx in order:
            score = float(pred_scores[idx])
            cat = int(pred_labels[idx])
            best_iou = 0.0
            best_gt = -1

            for gt_idx in range(len(gt_bboxes)):
                if gt_idx in matched_gt:
                    continue
                if int(gt_labels[gt_idx]) != cat:
                    continue
                if iou_matrix[idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[idx, gt_idx]
                    best_gt = gt_idx

            is_tp = best_iou >= iou_threshold and best_gt >= 0
            if is_tp:
                matched_gt.add(best_gt)

            self._det_records.append((score, is_tp, cat))

    def update(
        self,
        predictions: Dict,
        targets: Dict,
    ):
        """
        Convenience method that calls update_detection and optionally
        update_segmentation based on available keys.

        Args:
            predictions: dict with "bboxes" [N,4], "scores" [N], "labels" [N],
                         optionally "semantic_mask" [H,W].
            targets: dict with "bboxes" [M,4], "category_ids" [M],
                     optionally "semantic_mask" [H,W].
        """
        # Detection
        pred_bboxes = _to_numpy(predictions.get("bboxes", np.zeros((0, 4))))
        pred_scores = _to_numpy(predictions.get("scores", np.zeros(0)))
        pred_labels = _to_numpy(predictions.get("labels", np.zeros(0)))
        gt_bboxes = _to_numpy(targets.get("bboxes", np.zeros((0, 4))))
        gt_labels = _to_numpy(targets.get("category_ids", np.zeros(0)))

        self.update_detection(
            pred_bboxes, pred_scores, pred_labels,
            gt_bboxes, gt_labels,
            iou_threshold=self.cfg.f1_threshold,
        )

        # Segmentation
        if "semantic_mask" in predictions and "semantic_mask" in targets:
            self.update_segmentation(
                _to_numpy(predictions["semantic_mask"]),
                _to_numpy(targets["semantic_mask"]),
            )

    # ── Compute metrics ──────────────────────────────────────────────────

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.

        Returns dict with keys:
            miou, miou_per_class, f1, precision, recall,
            ap50, ap75, ap_5095
        """
        results = {}

        # ── mIoU ─────────────────────────────────────────────────────────
        results.update(self._compute_miou())

        # ── F1 / Precision / Recall ──────────────────────────────────────
        results.update(self._compute_f1())

        # ── AP @ multiple thresholds ─────────────────────────────────────
        results.update(self._compute_ap())

        return results

    def _compute_miou(self) -> Dict:
        """Compute mean IoU from the confusion matrix."""
        cm = self._confusion
        if cm.sum() == 0:
            return {"miou": 0.0, "miou_per_class": {}}

        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        union[union == 0] = 1  # avoid division by zero

        iou_per_class = intersection / union
        # Exclude background (class 0) from mIoU
        valid = iou_per_class[1:]  # classes 1..N
        miou = float(valid.mean()) if len(valid) > 0 else 0.0

        return {
            "miou": round(miou, 4),
            "miou_per_class": {
                i: round(float(v), 4) for i, v in enumerate(iou_per_class)
            },
        }

    def _compute_f1(self) -> Dict:
        """Compute detection F1, precision, recall."""
        if not self._det_records:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        tp = sum(1 for _, is_tp, _ in self._det_records if is_tp)
        fp = sum(1 for _, is_tp, _ in self._det_records if not is_tp)
        total_gt = sum(self._n_gt.values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    def _compute_ap(self) -> Dict:
        """Compute AP at IoU thresholds [0.5 : 0.05 : 0.95]."""
        if not self._det_records:
            return {"ap50": 0.0, "ap75": 0.0, "ap_5095": 0.0}

        # Re-compute for each threshold
        ap_values = {}
        for thr in self.cfg.iou_thresholds:
            ap_values[thr] = self._ap_at_threshold(thr)

        return {
            "ap50": round(ap_values.get(0.5, 0.0), 4),
            "ap75": round(ap_values.get(0.75, 0.0), 4),
            "ap_5095": round(float(np.mean(list(ap_values.values()))), 4),
        }

    def _ap_at_threshold(self, iou_threshold: float) -> float:
        """Compute AP for a single IoU threshold (11-point interpolation)."""
        # Sort by score descending
        records = sorted(self._det_records, key=lambda x: -x[0])
        total_gt = sum(self._n_gt.values())
        if total_gt == 0:
            return 0.0

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for score, is_tp, cat in records:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            prec = tp_cumsum / (tp_cumsum + fp_cumsum)
            rec = tp_cumsum / total_gt
            precisions.append(prec)
            recalls.append(rec)

        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(prec_at_rec) if prec_at_rec else 0.0
        ap /= 11.0

        return ap

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        union = area_a[:, None] + area_b[None, :] - inter
        union[union == 0] = 1

        return inter / union

    def summary(self) -> str:
        """Return a formatted string of all metrics."""
        results = self.compute()
        lines = [
            "=" * 50,
            "  EVALUATION RESULTS",
            "=" * 50,
            f"  mIoU:       {results['miou']:.4f}",
            f"  F1:         {results['f1']:.4f}",
            f"  Precision:  {results['precision']:.4f}",
            f"  Recall:     {results['recall']:.4f}",
            f"  AP@50:      {results['ap50']:.4f}",
            f"  AP@75:      {results['ap75']:.4f}",
            f"  AP@[.5:.95]:{results['ap_5095']:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ── Helper ───────────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    """Convert tensor / list to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)
