"""
MSNet Dataset Loader
====================
Xie & Xiong, Remote Sensing 2023 / Zhu et al. WACV 2021.
8,700+ images across 7 disaster events at 0.3–5 cm/px.
Multi-scale damage classification with oriented bounding boxes
and instance segmentation.

Expected directory layout (after download):
    datasets/msnet/
    ├── images/                     # RGB images
    ├── annotations/
    │   ├── instances_train.json    # COCO-format annotations
    │   ├── instances_val.json
    │   └── instances_test.json
    └── oriented_bboxes/            # (optional) oriented bbox DOTA format
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hurricane_debris.config import CATEGORY_QUERIES, MSNET_DAMAGE_MAP, DataConfig
from hurricane_debris.data.transforms import get_train_transforms, get_val_transforms
from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.msnet")

# MSNet native class names → our unified category IDs
_MSNET_CLASS_MAP = {
    "no-damage": 0,
    "minor-damage": 2,      # building_no_damage (low severity)
    "major-damage": 3,      # building_damaged
    "destroyed": 3,          # building_damaged
    "un-classified": 0,
    # Some MSNet variants use numeric IDs
}


class MSNetDataset(Dataset):
    """
    PyTorch Dataset for MSNet multi-scale disaster damage assessment.

    Supports:
      - Standard (axis-aligned) bounding boxes from COCO annotations
      - Instance segmentation masks where available
      - Oriented bounding boxes (converted to axis-aligned for SAM2 prompts)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        config: Optional[DataConfig] = None,
        task: str = "detection",
        annotation_file: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.config = config or DataConfig()
        self.split = split
        self.task = task
        self.image_size = self.config.image_size

        # ── Resolve annotation file ─────────────────────────────────────
        if annotation_file is None:
            annotation_file = f"annotations/instances_{split}.json"
        ann_path = self.root_dir / annotation_file

        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}. "
                "Please download the MSNet dataset."
            )

        with open(ann_path) as f:
            self.coco = json.load(f)

        # ── Build index ─────────────────────────────────────────────────
        self.images = self.coco.get("images", [])
        self.categories_raw = {
            c["id"]: c for c in self.coco.get("categories", [])
        }

        self.img_to_ann: Dict[int, List[Dict]] = {}
        for ann in self.coco.get("annotations", []):
            img_id = ann["image_id"]
            self.img_to_ann.setdefault(img_id, []).append(ann)

        # Keep images with annotations
        self.images = [
            img for img in self.images if img["id"] in self.img_to_ann
        ]

        logger.info(
            "MSNet [%s]: %d annotated images, %d annotations",
            split,
            len(self.images),
            sum(len(v) for v in self.img_to_ann.values()),
        )

        # ── Transforms ──────────────────────────────────────────────────
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

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img_info = self.images[idx]
        img_path = self.root_dir / "images" / img_info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Cannot read image: %s", img_path)
            return self._blank(idx)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        anns = self.img_to_ann.get(img_info["id"], [])

        bboxes, category_ids, masks = [], [], []

        for ann in anns:
            # Map MSNet category → unified taxonomy
            cat_id = self._map_category(ann)
            if cat_id == 0:
                continue  # skip background / no-damage

            bbox = ann.get("bbox")
            if bbox is None:
                # Try oriented bbox → axis-aligned conversion
                obbox = ann.get("oriented_bbox", ann.get("obbox"))
                if obbox is not None:
                    bbox = self._oriented_to_aabb(obbox)
                else:
                    continue

            # Validate
            x, y, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                continue
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            if bw <= 0 or bh <= 0:
                continue

            bboxes.append([x, y, bw, bh])
            category_ids.append(cat_id)

            # Instance mask (if present)
            if self.task in ("segmentation", "combined"):
                seg = ann.get("segmentation", [])
                if seg and isinstance(seg, list) and len(seg) > 0:
                    mask = self._polygons_to_mask(seg, h, w)
                    masks.append(mask)

        # ── Augment ─────────────────────────────────────────────────────
        try:
            extra = {}
            if bboxes:
                if masks and len(masks) == len(bboxes):
                    extra["masks"] = masks
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids,
                    **extra,
                )
            else:
                transformed = self.transform(
                    image=image, bboxes=[], category_ids=[]
                )
            image_t = transformed["image"]
            bboxes = list(transformed.get("bboxes", []))
            category_ids = list(transformed.get("category_ids", []))
            masks = transformed.get("masks", [])
        except Exception as e:
            logger.warning("Transform failed for %s: %s", img_path.name, e)
            return self._blank(idx)

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
                torch.stack([torch.from_numpy(m).float() for m in masks])
                if masks
                else torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
            ),
        }

        return {
            "pixel_values": image_t,
            "target": target,
            "image_id": img_info["id"],
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _map_category(self, ann: Dict) -> int:
        """Map an MSNet annotation to the unified category ID."""
        # Try numeric damage level first
        damage_level = ann.get("damage_level", ann.get("damage", None))
        if damage_level is not None and isinstance(damage_level, int):
            return MSNET_DAMAGE_MAP.get(damage_level, 0)

        # Try category name
        raw_cat_id = ann.get("category_id", 0)
        cat_info = self.categories_raw.get(raw_cat_id, {})
        cat_name = cat_info.get("name", "").lower().strip()
        return _MSNET_CLASS_MAP.get(cat_name, 0)

    @staticmethod
    def _oriented_to_aabb(obbox) -> List[float]:
        """Convert oriented bbox [cx, cy, w, h, angle] to axis-aligned [x, y, w, h]."""
        if len(obbox) == 5:
            cx, cy, ow, oh, angle = obbox
            # Rotate corners and find AABB
            cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            corners_x, corners_y = [], []
            for dx, dy in [(-ow / 2, -oh / 2), (ow / 2, -oh / 2),
                           (ow / 2, oh / 2), (-ow / 2, oh / 2)]:
                corners_x.append(cx + dx * cos_a - dy * sin_a)
                corners_y.append(cy + dx * sin_a + dy * cos_a)
            x1, y1 = min(corners_x), min(corners_y)
            x2, y2 = max(corners_x), max(corners_y)
            return [x1, y1, x2 - x1, y2 - y1]
        elif len(obbox) == 8:
            # Polygon corners: [x1,y1,x2,y2,x3,y3,x4,y4]
            xs = obbox[0::2]
            ys = obbox[1::2]
            x1, y1 = min(xs), min(ys)
            return [x1, y1, max(xs) - x1, max(ys) - y1]
        return obbox[:4]

    @staticmethod
    def _polygons_to_mask(polygons: List, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        return mask

    def _blank(self, idx: int) -> Dict:
        return {
            "pixel_values": torch.zeros(3, self.image_size, self.image_size),
            "target": {
                "bboxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": [],
                "category_ids": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros((0, self.image_size, self.image_size)),
            },
            "image_id": idx,
        }
