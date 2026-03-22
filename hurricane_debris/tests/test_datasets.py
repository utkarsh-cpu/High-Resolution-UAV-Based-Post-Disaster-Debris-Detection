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
from hurricane_debris.data.designsafe import DesignSafeDataset
from hurricane_debris.data.msnet import MSNetDataset
from hurricane_debris.data.rescuenet import RescueNetDataset
from hurricane_debris.data.transforms import get_train_transforms, stack_instance_masks


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

    def test_train_transforms_apply_with_current_albumentations_api(self):
        transform = get_train_transforms(image_size=128)
        image = np.zeros((256, 256, 3), dtype=np.uint8)

        transformed = transform(
            image=image,
            bboxes=[[32, 32, 64, 64]],
            category_ids=[3],
        )

        assert transformed["image"].shape == (3, 128, 128)
        assert len(transformed["bboxes"]) == 1

    def test_stack_instance_masks_accepts_tensor_masks_from_albumentations(self):
        transform = get_train_transforms(image_size=128)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[32:160, 32:160] = 1

        transformed = transform(
            image=image,
            bboxes=[[32, 32, 128, 128]],
            category_ids=[3],
            masks=[mask],
        )

        stacked_masks = stack_instance_masks(transformed["masks"], image_size=128)

        assert isinstance(transformed["masks"][0], torch.Tensor)
        assert stacked_masks.shape == (1, 128, 128)
        assert stacked_masks.dtype == torch.float32

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
        # Both should have spatial_transform
        assert type(ds_train.spatial_transform) is type(ds_val.spatial_transform)
        # But train has more transforms
        assert len(ds_train.spatial_transform.transforms) > len(ds_val.spatial_transform.transforms)

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


class TestDatasetSpecificSmoke:

    def test_rescuenet_loader_supports_official_lab_suffix_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_dir = root / "train" / "train-org-img"
            mask_dir = root / "train" / "train-label-img"
            img_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[16:48, 16:48] = 3

            cv2.imwrite(str(img_dir / "10778.jpg"), img)
            cv2.imwrite(str(mask_dir / "10778_lab.png"), mask)

            ds = RescueNetDataset(root_dir=str(root), split="train", config=DataConfig(image_size=64))
            assert len(ds) == 1
            sample = ds[0]
            assert sample["image_path"].endswith("10778.jpg")
            assert sample["target"]["bboxes"].shape[0] == 1

    def test_rescuenet_loader_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_dir = root / "test" / "test-org-img"
            mask_dir = root / "test" / "test-label-img"
            img_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[16:48, 16:48] = 3

            cv2.imwrite(str(img_dir / "sample.png"), img)
            cv2.imwrite(str(mask_dir / "sample.png"), mask)

            ds = RescueNetDataset(root_dir=str(root), split="test", config=DataConfig(image_size=64))
            assert len(ds) == 1
            sample = ds[0]
            assert "image_path" in sample
            assert sample["target"]["bboxes"].shape[1] == 4

    def test_rescuenet_loader_supports_dropbox_colourmask_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_dir = root / "RescueNet" / "test" / "test-org-img"
            mask_dir = root / "ColorMasks-RescueNet" / "test" / "test-label-img"
            img_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            mask_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            mask_rgb[16:48, 16:48] = (255, 0, 0)  # RescueNet official RGB for Building-Total-Destruction

            cv2.imwrite(str(img_dir / "sample.png"), img)
            cv2.imwrite(
                str(mask_dir / "sample.png"),
                cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR),
            )

            ds = RescueNetDataset(root_dir=str(root), split="test", config=DataConfig(image_size=64))
            assert len(ds) == 1
            sample = ds[0]
            assert "image_path" in sample
            assert sample["target"]["bboxes"].shape[0] == 1
            assert sample["target"]["category_ids"].tolist() == [3]

    def test_designsafe_loader_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "original").mkdir(parents=True)
            (root / "annotations").mkdir(parents=True)

            # Create a small RGB image and a grayscale mask with class 2
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "original" / "sample.png"), img)

            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[10:30, 10:30] = 2  # high-density debris
            cv2.imwrite(str(root / "annotations" / "sample.png"), mask)

            ds = DesignSafeDataset(root_dir=str(root), split="test", config=DataConfig(image_size=64))
            assert len(ds) >= 1
            sample = ds[0]
            assert "image_path" in sample
            assert sample["target"]["bboxes"].shape[1] == 4

    def test_msnet_loader_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir(parents=True)
            (root / "annotations").mkdir(parents=True)

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "images" / "sample.png"), img)

            coco = {
                "images": [{"id": 1, "file_name": "sample.png", "height": 64, "width": 64}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]}],
                "categories": [{"id": 2, "name": "major-damage"}],
            }
            with open(root / "annotations" / "instances_test.json", "w") as f:
                json.dump(coco, f)

            ds = MSNetDataset(root_dir=str(root), split="test", config=DataConfig(image_size=64))
            assert len(ds) == 1
            sample = ds[0]
            assert "image_path" in sample
            assert sample["target"]["bboxes"].shape[1] == 4

    def test_msnet_loader_supports_split_image_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "val").mkdir(parents=True)
            (root / "annotations").mkdir(parents=True)

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "val" / "sample.png"), img)

            coco = {
                "images": [{"id": 1, "file_name": "sample.png", "height": 64, "width": 64}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]}],
                "categories": [{"id": 2, "name": "major-damage"}],
            }
            with open(root / "annotations" / "instances_val.json", "w") as f:
                json.dump(coco, f)

            ds = MSNetDataset(root_dir=str(root), split="val", config=DataConfig(image_size=64))
            sample = ds[0]
            assert Path(sample["image_path"]).parent == root / "val"

    def test_msnet_loader_falls_back_from_test_to_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "val").mkdir(parents=True)
            (root / "annotations").mkdir(parents=True)

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "val" / "sample.png"), img)

            coco = {
                "images": [{"id": 1, "file_name": "sample.png", "height": 64, "width": 64}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]}],
                "categories": [{"id": 2, "name": "major-damage"}],
            }
            with open(root / "annotations" / "instances_val.json", "w") as f:
                json.dump(coco, f)

            ds = MSNetDataset(root_dir=str(root), split="test", config=DataConfig(image_size=64))
            assert ds.effective_split == "val"
            assert ds.annotation_path == root / "annotations" / "instances_val.json"
            sample = ds[0]
            assert Path(sample["image_path"]).parent == root / "val"
