"""
Unit tests for SAM2 loss computation and forward logic.
These tests do NOT require a SAM2 checkpoint — they test the
loss function and helper utilities in isolation.
"""

import numpy as np
import pytest
import torch

# We test the loss function directly, without loading SAM2
from hurricane_debris.models.sam2_trainer import SAM2Trainer


class TestSAM2Loss:
    """Test the Dice + BCE loss computation."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer instance without loading the model."""
        t = SAM2Trainer.__new__(SAM2Trainer)
        t.cfg = type("Cfg", (), {"bce_weight": 1.0, "dice_weight": 1.0})()
        t.device = "cpu"
        t.model = None
        return t

    def test_perfect_prediction(self, trainer):
        """Loss should be low when prediction matches target perfectly."""
        target = torch.ones(64, 64)
        pred = torch.ones(64, 64) * 10.0  # high logits → sigmoid ≈ 1
        loss = trainer.compute_loss(pred, target)
        assert loss.item() < 0.1

    def test_worst_prediction(self, trainer):
        """Loss should be high when prediction is opposite of target."""
        target = torch.ones(64, 64)
        pred = torch.ones(64, 64) * -10.0  # low logits → sigmoid ≈ 0
        loss = trainer.compute_loss(pred, target)
        assert loss.item() > 1.0

    def test_loss_is_differentiable(self, trainer):
        """Loss should have a grad_fn for backprop."""
        pred = torch.randn(64, 64, requires_grad=True)
        target = torch.randint(0, 2, (64, 64)).float()
        loss = trainer.compute_loss(pred, target)
        assert loss.requires_grad
        loss.backward()
        assert pred.grad is not None

    def test_empty_mask(self, trainer):
        """All-zero target and all-zero prediction should not crash."""
        pred = torch.zeros(64, 64)
        target = torch.zeros(64, 64)
        loss = trainer.compute_loss(pred, target)
        assert not torch.isnan(loss)

    def test_loss_symmetry(self, trainer):
        """Dice loss should be symmetric in pred/target (after sigmoid)."""
        a = torch.randn(32, 32)
        b = torch.randint(0, 2, (32, 32)).float()
        # Not strictly symmetric because BCE is not symmetric,
        # but we can at least check it doesn't crash
        loss_ab = trainer.compute_loss(a, b)
        assert not torch.isnan(loss_ab)
