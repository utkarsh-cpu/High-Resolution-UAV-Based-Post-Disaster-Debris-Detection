"""
Unit tests for Florence-2 label generation.
These are fast tests that do NOT load the actual model.
"""

from contextlib import contextmanager

import numpy as np
import pytest
import torch

from hurricane_debris.models import florence2 as florence2_module
from hurricane_debris.models.florence2 import (
    Florence2Trainer,
    _bbox_coco_to_florence,
    load_florence_processor,
)


class TestFlorence2Labels:

    def test_load_florence_processor_retries_with_slow_tokenizer(self, monkeypatch):
        calls = []
        expected_processor = object()

        def fake_from_pretrained(model_id, trust_remote_code=True, use_fast=True):
            calls.append({
                "model_id": model_id,
                "trust_remote_code": trust_remote_code,
                "use_fast": use_fast,
            })
            if use_fast:
                raise AttributeError(
                    "TokenizersBackend has no attribute additional_special_tokens"
                )
            return expected_processor

        monkeypatch.setattr(
            florence2_module.AutoProcessor,
            "from_pretrained",
            fake_from_pretrained,
        )

        processor = load_florence_processor("microsoft/Florence-2-base-ft")

        assert processor is expected_processor
        assert calls == [
            {
                "model_id": "microsoft/Florence-2-base-ft",
                "trust_remote_code": True,
                "use_fast": True,
            },
            {
                "model_id": "microsoft/Florence-2-base-ft",
                "trust_remote_code": True,
                "use_fast": False,
            },
        ]

    def test_bbox_to_florence_basic(self):
        """A bbox at (0, 0, 100, 100) in a 1000×1000 image → <loc_0><loc_0><loc_100><loc_100>."""
        result = _bbox_coco_to_florence([0, 0, 100, 100], img_w=1000, img_h=1000)
        assert "<loc_0>" in result
        # x2 = 100/1000 * 999 ≈ 100
        assert result.count("<loc_") == 4

    def test_bbox_to_florence_full_image(self):
        """A bbox covering the full image should produce <loc_0><loc_0><loc_999><loc_999>."""
        result = _bbox_coco_to_florence([0, 0, 1000, 1000], img_w=1000, img_h=1000)
        assert result == "<loc_0><loc_0><loc_999><loc_999>"

    def test_bbox_to_florence_clamping(self):
        """Coordinates exceeding the image should be clamped to 999."""
        result = _bbox_coco_to_florence([0, 0, 2000, 2000], img_w=1000, img_h=1000)
        assert "<loc_999>" in result

    def test_bbox_to_florence_center(self):
        """A bbox in the center of the image."""
        result = _bbox_coco_to_florence([250, 250, 500, 500], img_w=1000, img_h=1000)
        # x1 = 250/1000*999 ≈ 250, y1 ≈ 250, x2 = 750/1000*999 ≈ 749, y2 ≈ 749
        assert "<loc_250>" in result

    def test_output_format(self):
        """Output should consist of exactly 4 <loc_N> tokens."""
        result = _bbox_coco_to_florence([10, 20, 30, 40], img_w=100, img_h=100)
        tokens = result.split("><")
        assert len(tokens) == 4  # 4 loc tokens

    def test_collate_uses_detection_targets_for_labels(self):
        class DummyBatch(dict):
            def to(self, _device):
                return self

        class DummyTokenizer:
            pad_token_id = 0

            def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=256):
                # One tokenized target sequence per text.
                rows = []
                for i, _ in enumerate(texts):
                    rows.append([5 + i, 9, 0])
                return {"input_ids": torch.tensor(rows, dtype=torch.long)}

        class DummyProcessor:
            def __init__(self):
                self.tokenizer = DummyTokenizer()

            @contextmanager
            def as_target_processor(self):
                yield self

            def __call__(self, text, images, return_tensors=None, padding=True):
                # Prompt token ids intentionally different from target token ids.
                return DummyBatch(
                    {
                        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long),
                        "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
                    }
                )

        trainer = Florence2Trainer.__new__(Florence2Trainer)
        trainer.device = "cpu"
        trainer.cfg = type("Cfg", (), {"max_new_tokens": 32})()
        trainer.processor = DummyProcessor()

        sample = {
            "pixel_values": torch.zeros((3, 8, 8), dtype=torch.float32),
            "target": {
                "bboxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]], dtype=torch.float32),
                "labels": ["vehicle"],
            },
        }

        batch = trainer.collate_fn([sample])
        assert "labels" in batch
        assert not torch.equal(batch["labels"], batch["input_ids"])
