"""
Unit tests for Florence-2 label generation.
These are fast tests that do NOT load the actual model.
"""

import numpy as np
import pytest
import torch

from hurricane_debris.models.florence2 import _bbox_coco_to_florence


class TestFlorence2Labels:

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
