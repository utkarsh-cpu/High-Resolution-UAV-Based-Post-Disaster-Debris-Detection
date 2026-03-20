"""
Base PyTorch Dataset for hurricane debris detection.
All dataset-specific loaders inherit from this class.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hurricane_debris.config import CATEGORY_QUERIES, DataConfig
from hurricane_debris.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    stack_instance_masks,
)
from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.base")


class DebrisDataset(Dataset):
    """
    Unified PyTorch Dataset for debris detection / segmentation.

    Expects COCO-format annotations and supports:
      - Detection (bboxes + category labels)
      - Segmentation (polygon/mask ground truth)
      - Combined mode (both)

    Concrete datasets (RescueNet, MSNet, …) should override
    ``_load_annotations`` and ``_build_coco_annotations`` to convert
    their native format into the COCO dict consumed here.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str = "annotations.json",
        split: str = "train",
        image_ids: Optional[List[int]] = None,
        config: Optional[DataConfig] = None,
        task: str = "detection",  # "detection" | "segmentation" | "combined"
    ):
        self.root_dir = Path(root_dir)
        self.config = config or DataConfig()
        self.split = split
        self.task = task
        self.image_size = self.config.image_size

        # ── Load annotations ────────────────────────────────────────────
        ann_path = self.root_dir / annotation_file
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path) as f:
            self.data = json.load(f)

        self.categories = {c["id"]: c for c in self.data.get("categories", [])}
        self._build_index()

        # ── Filter to requested split (by image ID list) ────────────────
        if image_ids is not None:
            id_set = set(image_ids)
            self.images = [img for img in self.images if img["id"] in id_set]
            logger.info(
                "Filtered to %d images for split='%s'", len(self.images), split
            )

        # ── Augmentation pipeline ───────────────────────────────────────
        if split == "train" and self.config.augment_train:
            self.transform = get_train_transforms(
                image_size=self.image_size,
                crop_scale=self.config.random_crop_scale,
                mean=self.config.image_mean,
                std=self.config.image_std,
                color_jitter_p=self.config.color_jitter_p,
                gauss_noise_p=self.config.gauss_noise_p,
            )
        else:
            self.transform = get_val_transforms(
                image_size=self.image_size,
                mean=self.config.image_mean,
                std=self.config.image_std,
            )

    # ── Index helpers ────────────────────────────────────────────────────

    def _build_index(self):
        """Build image list and image → annotation mapping."""
        self.images: List[Dict] = self.data.get("images", [])
        annotations = self.data.get("annotations", [])

        self.img_to_ann: Dict[int, List[Dict]] = defaultdict(list)
        for ann in annotations:
            self.img_to_ann[ann["image_id"]].append(ann)

        # Keep only images that have at least one annotation
        self.images = [
            img for img in self.images if img["id"] in self.img_to_ann
        ]
        logger.info("Loaded %d annotated images", len(self.images))

    # ── Core Dataset interface ───────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img_info = self.images[idx]
        img_path = self.root_dir / "images" / img_info["file_name"]

        # ── Load image with error handling ──────────────────────────────
        image = self._safe_load_image(img_path)
        if image is None:
            # Return a blank sample so the DataLoader doesn't crash
            return self._blank_sample(img_info["id"])

        # ── Parse annotations ───────────────────────────────────────────
        anns = self.img_to_ann.get(img_info["id"], [])
        bboxes, category_ids, masks = self._parse_annotations(
            anns, img_info.get("height", image.shape[0]),
            img_info.get("width", image.shape[1]),
        )

        # ── Validate bboxes ────────────────────────────────────────────
        bboxes, category_ids, masks = self._validate_bboxes(
            bboxes, category_ids, masks, image.shape[1], image.shape[0]
        )

        # ── Apply augmentations ─────────────────────────────────────────
        try:
            if len(bboxes) > 0:
                extra = {"masks": masks} if masks else {}
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids,
                    **extra,
                )
                image_t = transformed["image"]
                bboxes = list(transformed["bboxes"])
                category_ids = list(transformed["category_ids"])
                masks = transformed.get("masks", [])
            else:
                transformed = self.transform(
                    image=image, bboxes=[], category_ids=[]
                )
                image_t = transformed["image"]
        except Exception as e:
            logger.warning("Transform failed for image %s: %s", img_path.name, e)
            return self._blank_sample(img_info["id"])

        # ── Build target dict ───────────────────────────────────────────
        text_queries = [
            CATEGORY_QUERIES.get(cid, "debris") for cid in category_ids
        ]

        target = {
            "bboxes": (
                torch.tensor(bboxes, dtype=torch.float32)
                if bboxes
                else torch.zeros((0, 4), dtype=torch.float32)
            ),
            "labels": text_queries,
            "category_ids": (
                torch.tensor(category_ids, dtype=torch.long)
                if category_ids
                else torch.zeros(0, dtype=torch.long)
            ),
            "masks": stack_instance_masks(masks, self.image_size),
        }

        return {
            "pixel_values": image_t,
            "target": target,
            "image_id": img_info["id"],
            "image_path": str(img_path),
        }

    # ── Private helpers ──────────────────────────────────────────────────

    def _safe_load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load an image from disk with error handling."""
        if not path.exists():
            logger.warning("Image not found: %s", path)
            return None
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Failed to decode image: %s", path)
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _parse_annotations(
        self,
        anns: List[Dict],
        height: int,
        width: int,
    ) -> Tuple[List, List, List]:
        """Extract bboxes, category IDs, and masks from annotations."""
        bboxes, category_ids, masks = [], [], []
        for ann in anns:
            bboxes.append(ann["bbox"])  # COCO format [x, y, w, h]
            category_ids.append(ann["category_id"])

            if self.task in ("segmentation", "combined"):
                seg = ann.get("segmentation", [])
                if seg:
                    mask = self._polygons_to_mask(seg, height, width)
                    masks.append(mask)

        return bboxes, category_ids, masks if masks else []

    @staticmethod
    def _polygons_to_mask(
        polygons: List, height: int, width: int
    ) -> np.ndarray:
        """Convert COCO polygon to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        return mask

    @staticmethod
    def _validate_bboxes(
        bboxes: List,
        category_ids: List,
        masks: List,
        img_w: int,
        img_h: int,
    ) -> Tuple[List, List, List]:
        """Filter out invalid bboxes (zero area, out of bounds)."""
        valid_b, valid_c, valid_m = [], [], []
        for i, (bbox, cid) in enumerate(zip(bboxes, category_ids)):
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            # Clamp to image bounds
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            if w <= 0 or h <= 0:
                continue
            valid_b.append([x, y, w, h])
            valid_c.append(cid)
            if masks and i < len(masks):
                valid_m.append(masks[i])
        return valid_b, valid_c, valid_m

    def _blank_sample(self, image_id: int) -> Dict:
        """Return a zero-filled sample as fallback."""
        return {
            "pixel_values": torch.zeros(
                3, self.image_size, self.image_size, dtype=torch.float32
            ),
            "target": {
                "bboxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": [],
                "category_ids": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros(
                    (0, self.image_size, self.image_size), dtype=torch.float32
                ),
            },
            "image_id": image_id,
            "image_path": "",
        }
