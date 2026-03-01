"""
Unit tests for dataset loaders.
Run with:  pytest hurricane_debris/tests/ -v
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from hurricane_debris.config import DataConfig
from hurricane_debris.data.base_dataset import DebrisDataset


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_dataset_dir():
    """Create a temporary dataset with 5 images and COCO annotations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()

        images = []
        annotations = []
        ann_id = 0

        for i in range(5):
            # Create 256×256 dummy image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            fname = f"test_{i:03d}.png"
            cv2.imwrite(str(img_dir / fname), img)

            images.append({
                "id": i,
                "file_name": fname,
                "height": 256,
                "width": 256,
            })

            # 2 annotations per image
            for j in range(2):
                x = np.random.randint(10, 100)
                y = np.random.randint(10, 100)
                w = np.random.randint(30, 80)
                h = np.random.randint(30, 80)
                cat_id = np.random.randint(1, 8)

                annotations.append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": int(cat_id),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(w * h),
                    "segmentation": [[
                        int(x), int(y),
                        int(x + w), int(y),
                        int(x + w), int(y + h),
                        int(x), int(y + h),
                    ]],
                    "iscrowd": 0,
                })
                ann_id += 1

        categories = [
            {"id": i, "name": f"class_{i}", "supercategory": "test"}
            for i in range(8)
        ]

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(Path(tmpdir) / "annotations.json", "w") as f:
            json.dump(coco, f)

        yield tmpdir


# ── Tests ────────────────────────────────────────────────────────────────


class TestDebrisDataset:

    def test_length(self, dummy_dataset_dir):
        ds = DebrisDataset(dummy_dataset_dir, split="val")
        assert len(ds) == 5

    def test_getitem_returns_correct_keys(self, dummy_dataset_dir):
        ds = DebrisDataset(dummy_dataset_dir, split="val", config=DataConfig(image_size=128))
        sample = ds[0]
        assert "pixel_values" in sample
        assert "target" in sample
        assert "image_id" in sample

    def test_pixel_values_shape(self, dummy_dataset_dir):
        cfg = DataConfig(image_size=128)
        ds = DebrisDataset(dummy_dataset_dir, split="val", config=cfg)
        sample = ds[0]
        assert sample["pixel_values"].shape == (3, 128, 128)

    def test_target_keys(self, dummy_dataset_dir):
        ds = DebrisDataset(dummy_dataset_dir, split="val")
        target = ds[0]["target"]
        assert "bboxes" in target
        assert "labels" in target
        assert "category_ids" in target
        assert "masks" in target

    def test_bboxes_shape(self, dummy_dataset_dir):
        ds = DebrisDataset(dummy_dataset_dir, split="val")
        bboxes = ds[0]["target"]["bboxes"]
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4

    def test_category_ids_dtype(self, dummy_dataset_dir):
        ds = DebrisDataset(dummy_dataset_dir, split="val")
        cat_ids = ds[0]["target"]["category_ids"]
        assert cat_ids.dtype == torch.long

    def test_labels_are_strings(self, dummy_dataset_dir):
        ds = DebrisDataset(dummy_dataset_dir, split="val")
        labels = ds[0]["target"]["labels"]
        assert isinstance(labels, list)
        if labels:
            assert isinstance(labels[0], str)

    def test_train_augmentation_differs(self, dummy_dataset_dir):
        cfg = DataConfig(image_size=128)
        ds_train = DebrisDataset(dummy_dataset_dir, split="train", config=cfg)
        ds_val = DebrisDataset(dummy_dataset_dir, split="val", config=cfg)
        # Augmentation pipelines should be different objects
        assert type(ds_train.transform) is type(ds_val.transform)
        # But train has more transforms
        assert len(ds_train.transform.transforms) > len(ds_val.transform.transforms)

    def test_blank_sample_on_missing_image(self, dummy_dataset_dir):
        """Dataset should return blank sample instead of crashing."""
        ds = DebrisDataset(dummy_dataset_dir, split="val")
        # Corrupt the path
        ds.images[0]["file_name"] = "nonexistent.png"
        sample = ds[0]
        # Should still return valid structure
        assert sample["pixel_values"].shape[0] == 3
        assert len(sample["target"]["labels"]) == 0

    def test_image_id_filtering(self, dummy_dataset_dir):
        """Filtering by image_ids should reduce dataset length."""
        ds = DebrisDataset(
            dummy_dataset_dir, split="val", image_ids=[0, 2]
        )
        assert len(ds) == 2
