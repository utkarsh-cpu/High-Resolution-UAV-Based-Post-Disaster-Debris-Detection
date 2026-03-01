"""
DesignSafe-CI PRJ-6029 Dataset Loader
======================================
NSF NHERI DesignSafe-CI Repository.
Multi-hazard UAV and ground-based imagery from earthquakes, hurricanes, and floods.
Used for **cross-dataset generalisation testing** (not primary training).

DOI: 10.17603/ds2-jvps-2n95

Expected directory layout (after download):
    datasets/designsafe/
    ├── images/                 # All image files
    ├── annotations/
    │   └── damage_observations.json   # Georeferenced damage metadata
    └── splits/
        ├── train.txt           # (Optional) official splits
        ├── val.txt
        └── test.txt
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hurricane_debris.config import CATEGORY_QUERIES, DataConfig
from hurricane_debris.data.splits import load_official_split
from hurricane_debris.data.transforms import get_train_transforms, get_val_transforms
from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.designsafe")

# Mapping of DesignSafe hazard-damage labels → unified category IDs
_HAZARD_TO_CATEGORY = {
    "building_damage": 3,    # building_damaged
    "structural_damage": 3,
    "roof_damage": 3,
    "flood": 1,              # water
    "flooding": 1,
    "road_damage": 6,        # road_damaged
    "vegetation_damage": 4,  # vegetation
    "vehicle_damage": 7,     # vehicle
    "debris": 3,             # default to building_damaged
}


class DesignSafeDataset(Dataset):
    """
    PyTorch Dataset for DesignSafe-CI multi-hazard imagery.

    Primarily intended for **evaluation / cross-dataset testing**.
    The dataset contains georeferenced damage observations that we convert
    to bounding-box annotations using the provided coordinates.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        config: Optional[DataConfig] = None,
        task: str = "detection",
        annotation_file: str = "annotations/damage_observations.json",
    ):
        self.root_dir = Path(root_dir)
        self.config = config or DataConfig()
        self.split = split
        self.task = task
        self.image_size = self.config.image_size

        # ── Load annotations ────────────────────────────────────────────
        self.ann_path = self.root_dir / annotation_file
        if not self.ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.ann_path}. "
                "Please download the DesignSafe-CI dataset and convert annotations."
            )

        with open(self.ann_path) as f:
            raw = json.load(f)

        # Normalise to COCO-like structure
        self.images, self.img_to_ann = self._parse_annotations(raw)

        # ── Apply official split if available ────────────────────────────
        split_files = load_official_split(
            str(self.root_dir / "splits"), split
        )
        if split_files is not None:
            allowed = set(split_files)
            self.images = [
                img for img in self.images if img["file_name"] in allowed
            ]
        logger.info(
            "DesignSafe [%s]: %d images loaded", split, len(self.images)
        )

        # ── Transforms ──────────────────────────────────────────────────
        # Always use val transforms for this test-oriented dataset
        self.transform = get_val_transforms(
            image_size=self.image_size,
            mean=self.config.image_mean,
            std=self.config.image_std,
        )

    # ── Annotation parsing ───────────────────────────────────────────────

    def _parse_annotations(self, raw: dict) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
        """
        Convert DesignSafe JSON to a COCO-like internal representation.

        Supports two formats:
          A) Already COCO-formatted (has "images" and "annotations" keys)
          B) Custom list-of-observations format
        """
        # Format A: standard COCO
        if "images" in raw and "annotations" in raw:
            images = raw["images"]
            img_to_ann: Dict[int, List[Dict]] = {}
            for ann in raw["annotations"]:
                img_id = ann["image_id"]
                img_to_ann.setdefault(img_id, []).append(ann)
            return images, img_to_ann

        # Format B: list of observation dicts
        images = []
        img_to_ann = {}
        seen_files: Dict[str, int] = {}
        ann_id = 0

        observations = raw if isinstance(raw, list) else raw.get("observations", [])
        for obs in observations:
            fname = obs.get("image", obs.get("file_name", ""))
            if not fname:
                continue

            if fname not in seen_files:
                img_id = len(seen_files)
                seen_files[fname] = img_id
                images.append({
                    "id": img_id,
                    "file_name": fname,
                    "height": obs.get("height", 1024),
                    "width": obs.get("width", 1024),
                })
            else:
                img_id = seen_files[fname]

            # Map hazard type → category
            hazard = obs.get("hazard_type", obs.get("damage_type", "debris"))
            cat_id = _HAZARD_TO_CATEGORY.get(hazard.lower(), 3)

            bbox = obs.get("bbox", None)
            if bbox is None:
                continue

            img_to_ann.setdefault(img_id, []).append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,  # [x, y, w, h]
                "area": bbox[2] * bbox[3] if len(bbox) == 4 else 0,
                "segmentation": obs.get("segmentation", []),
                "iscrowd": 0,
            })
            ann_id += 1

        return images, img_to_ann

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

        anns = self.img_to_ann.get(img_info["id"], [])
        bboxes = [a["bbox"] for a in anns]
        category_ids = [a["category_id"] for a in anns]

        try:
            if bboxes:
                transformed = self.transform(
                    image=image, bboxes=bboxes, category_ids=category_ids
                )
            else:
                transformed = self.transform(
                    image=image, bboxes=[], category_ids=[]
                )
            image_t = transformed["image"]
            bboxes = list(transformed.get("bboxes", []))
            category_ids = list(transformed.get("category_ids", []))
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
            "masks": torch.zeros(
                (0, self.image_size, self.image_size), dtype=torch.float32
            ),
        }

        return {
            "pixel_values": image_t,
            "target": target,
            "image_id": img_info["id"],
        }

    def _blank(self, idx: int) -> Dict:
        return {
            "pixel_values": torch.zeros(3, self.image_size, self.image_size),
            "target": {
                "bboxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": [],
                "category_ids": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros(
                    (0, self.image_size, self.image_size), dtype=torch.float32
                ),
            },
            "image_id": idx,
        }
