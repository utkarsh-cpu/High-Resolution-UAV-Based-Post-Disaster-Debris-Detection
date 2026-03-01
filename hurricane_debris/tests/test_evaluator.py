"""
Unit tests for the Evaluator class.
"""

import numpy as np
import pytest

from hurricane_debris.config import EvalConfig
from hurricane_debris.evaluation.metrics import Evaluator


class TestEvaluator:

    @pytest.fixture
    def evaluator(self):
        return Evaluator(config=EvalConfig(num_classes=3))

    # ── mIoU tests ────────────────────────────────────────────────────────

    def test_perfect_segmentation(self, evaluator):
        mask = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        evaluator.update_segmentation(mask, mask)
        results = evaluator.compute()
        assert results["miou"] == 1.0

    def test_wrong_segmentation(self, evaluator):
        pred = np.zeros((10, 10), dtype=int)
        gt = np.ones((10, 10), dtype=int)
        evaluator.update_segmentation(pred, gt)
        results = evaluator.compute()
        assert results["miou"] < 0.5

    def test_empty_confusion(self, evaluator):
        results = evaluator.compute()
        assert results["miou"] == 0.0

    # ── F1 / detection tests ─────────────────────────────────────────────

    def test_perfect_detection(self, evaluator):
        pred_boxes = np.array([[10, 10, 50, 50]], dtype=float)
        pred_scores = np.array([0.9])
        pred_labels = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=float)
        gt_labels = np.array([1])

        evaluator.update_detection(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 0.5
        )
        results = evaluator.compute()
        assert results["f1"] == 1.0
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0

    def test_no_predictions(self, evaluator):
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=float)
        gt_labels = np.array([1])

        evaluator.update_detection(
            np.zeros((0, 4)), np.zeros(0), np.zeros(0),
            gt_boxes, gt_labels, 0.5,
        )
        results = evaluator.compute()
        assert results["recall"] == 0.0

    def test_false_positive(self, evaluator):
        pred_boxes = np.array([[10, 10, 50, 50]], dtype=float)
        pred_scores = np.array([0.9])
        pred_labels = np.array([1])

        evaluator.update_detection(
            pred_boxes, pred_scores, pred_labels,
            np.zeros((0, 4)), np.zeros(0, dtype=int), 0.5,
        )
        results = evaluator.compute()
        assert results["precision"] == 0.0

    # ── Reset ────────────────────────────────────────────────────────────

    def test_reset(self, evaluator):
        mask = np.ones((5, 5), dtype=int)
        evaluator.update_segmentation(mask, mask)
        evaluator.reset()
        results = evaluator.compute()
        assert results["miou"] == 0.0

    # ── Summary string ───────────────────────────────────────────────────

    def test_summary(self, evaluator):
        summary = evaluator.summary()
        assert "mIoU" in summary
        assert "F1" in summary
        assert "AP@50" in summary
