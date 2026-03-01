"""
RescueNet Dataset Loader
========================
Alam et al., IEEE TGRS 2022.
4,494 images at 0.5–2 cm/px GSD from Hurricane Michael (FL, 2018).
8-class pixel-level semantic segmentation with official train/val/test splits.

Expected directory layout (after download):
    datasets/rescuenet/
    ├── train/
    │   ├── train-org-img/      # RGB images
    │   └── train-label-img/    # Semantic masks (pixel value = class ID)
    ├── val/
    │   ├── val-org-img/
    │   └── val-label-img/
    └── test/
        ├── test-org-img/
        └── test-label-img/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hurricane_debris.config import (
    CATEGORY_QUERIES,
    DEBRIS_CATEGORIES,
    RESCUENET_CLASS_MAP,
    DataConfig,
)
from hurricane_debris.data.transforms import get_train_transforms, get_val_transforms
from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.rescuenet")

# The folder naming convention inside each split
_SPLIT_DIRS = {
    "train": ("train/train-org-img", "train/train-label-img"),
    "val": ("val/val-org-img", "val/val-label-img"),
    "test": ("test/test-org-img", "test/test-label-img"),
}


class RescueNetDataset(Dataset):
    """
    PyTorch Dataset for RescueNet semantic segmentation.

    Each sample returns:
      - pixel_values: augmented image tensor  [3, H, W]
      - target.bboxes: bounding boxes derived from connected components [N, 4]
      - target.labels: text queries per object
      - target.category_ids: class IDs  [N]
      - target.masks: per-instance binary masks  [N, H, W]
      - target.semantic_mask: full semantic mask  [H, W]
      - image_id: unique identifier
    """

    # Classes as defined by the RescueNet benchmark
    CLASS_NAMES = [
        "background",
        "water",
        "building_no_damage",
        "building_damaged",
        "vegetation",
        "road_no_damage",
        "road_damaged",
        "vehicle",
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        config: Optional[DataConfig] = None,
        task: str = "segmentation",
        min_component_area: int = 100,
    ):
        """
        Args:
            root_dir: Path to rescuenet/ directory.
            split: One of "train", "val", "test".
            config: Data configuration.
            task: "detection", "segmentation", or "combined".
            min_component_area: Minimum pixel area for a connected component
                                to be treated as an instance.
        """
        self.root_dir = Path(root_dir)
        self.config = config or DataConfig()
        self.split = split
        self.task = task
        self.image_size = self.config.image_size
        self.min_area = min_component_area

        if split not in _SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(_SPLIT_DIRS)}, got '{split}'")

        img_subdir, mask_subdir = _SPLIT_DIRS[split]
        self.img_dir = self.root_dir / img_subdir
        self.mask_dir = self.root_dir / mask_subdir

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}. "
                "Please download RescueNet and extract it following the expected layout."
            )

        # Gather paired image/mask paths
        self.samples: List[Tuple[Path, Path]] = self._discover_samples()
        logger.info(
            "RescueNet [%s]: %d samples from %s",
            split, len(self.samples), self.img_dir,
        )

        # Transforms
        if split == "train" and self.config.augment_train:
            self.transform = get_train_transforms(
                image_size=self.image_size,
                crop_scale=self.config.random_crop_scale,
                mean=self.config.image_mean,
                std=self.config.image_std,
            )
        else:
            self.transform = get_val_transforms(
                image_size=self.image_size,
                mean=self.config.image_mean,
                std=self.config.image_std,
            )

    # ── Discovery ────────────────────────────────────────────────────────

    def _discover_samples(self) -> List[Tuple[Path, Path]]:
        """Find all (image, mask) pairs that exist on disk."""
        samples = []
        img_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in img_extensions:
                continue
            # Mask file typically has the same stem
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                # Try .png if image is .jpg
                mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                samples.append((img_path, mask_path))
            else:
                logger.warning("No mask found for %s", img_path.name)
        return samples

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, mask_path = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Cannot read image: %s", img_path)
            return self._blank_sample(idx)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load semantic mask (pixel value = class ID)
        semantic_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if semantic_mask is None:
            logger.warning("Cannot read mask: %s", mask_path)
            return self._blank_sample(idx)

        # Remap classes through the unified taxonomy
        semantic_mask = self._remap_classes(semantic_mask)

        # Derive per-instance bboxes + masks from connected components
        bboxes, category_ids, instance_masks = self._mask_to_instances(
            semantic_mask, image.shape[1], image.shape[0]
        )

        # ── Apply augmentations ──────────────────────────────────────────
        try:
            if bboxes:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids,
                    masks=instance_masks,
                )
                image_t = transformed["image"]
                bboxes = list(transformed["bboxes"])
                category_ids = list(transformed["category_ids"])
                instance_masks = transformed.get("masks", [])
            else:
                transformed = self.transform(
                    image=image, bboxes=[], category_ids=[]
                )
                image_t = transformed["image"]
        except Exception as e:
            logger.warning("Transform failed for %s: %s", img_path.name, e)
            return self._blank_sample(idx)

        # Resize semantic mask to target size for evaluation
        sem_mask_resized = cv2.resize(
            semantic_mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        text_queries = [
            CATEGORY_QUERIES.get(cid, "debris") for cid in category_ids
        ]

        target = {
            "bboxes": (
                torch.tensor(bboxes, dtype=torch.float32)
                if bboxes else torch.zeros((0, 4), dtype=torch.float32)
            ),
            "labels": text_queries,
            "category_ids": (
                torch.tensor(category_ids, dtype=torch.long)
                if category_ids else torch.zeros(0, dtype=torch.long)
            ),
            "masks": (
                torch.stack([torch.from_numpy(m).float() for m in instance_masks])
                if instance_masks
                else torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
            ),
            "semantic_mask": torch.from_numpy(sem_mask_resized).long(),
        }

        return {
            "pixel_values": image_t,
            "target": target,
            "image_id": idx,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _remap_classes(mask: np.ndarray) -> np.ndarray:
        """Remap RescueNet class IDs through the unified taxonomy."""
        out = np.zeros_like(mask)
        for src, dst in RESCUENET_CLASS_MAP.items():
            out[mask == src] = dst
        return out

    def _mask_to_instances(
        self, semantic_mask: np.ndarray, img_w: int, img_h: int
    ) -> Tuple[List, List, List]:
        """
        Derive instance bboxes and binary masks from a semantic mask
        using connected-component analysis.
        """
        bboxes, cat_ids, inst_masks = [], [], []

        for class_id in range(1, len(self.CLASS_NAMES)):  # skip background
            binary = (semantic_mask == class_id).astype(np.uint8)
            if binary.sum() == 0:
                continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            for label_idx in range(1, num_labels):  # skip background component
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area < self.min_area:
                    continue
                x = stats[label_idx, cv2.CC_STAT_LEFT]
                y = stats[label_idx, cv2.CC_STAT_TOP]
                w = stats[label_idx, cv2.CC_STAT_WIDTH]
                h = stats[label_idx, cv2.CC_STAT_HEIGHT]

                inst_mask = (labels == label_idx).astype(np.uint8)
                bboxes.append([x, y, w, h])
                cat_ids.append(class_id)
                inst_masks.append(inst_mask)

        return bboxes, cat_ids, inst_masks

    def _blank_sample(self, idx: int) -> Dict:
        return {
            "pixel_values": torch.zeros(3, self.image_size, self.image_size),
            "target": {
                "bboxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": [],
                "category_ids": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros((0, self.image_size, self.image_size)),
                "semantic_mask": torch.zeros(
                    self.image_size, self.image_size, dtype=torch.long
                ),
            },
            "image_id": idx,
        }
