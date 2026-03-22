"""
DesignSafe-CI PRJ-6029 Dataset Loader
======================================
Amini et al., "Debris Segmentation using Post-Hurricane Aerial Imagery."
NSF NHERI DesignSafe-CI Repository.  DOI: 10.17603/ds2-jvps-2n95

Hurricane-induced debris segmentation from aerial imagery (Ian, Ida, Ike).
1,242 images total, 508 with pixel-level debris masks.

Annotation classes:
    0 = No debris
    1 = Low-density debris
    2 = High-density debris

Expected directory layout (after download):
    datasets/designsafe/
    └── PRJ-6029/
        └── Project--hurricane-induced-debris-segmentation-dataset-using-aerial-imagery--V2/
            └── data/
                ├── original/            # 1,242 RGB aerial image crops
                ├── annotations/         # 508 grayscale segmentation masks
                ├── annotations_vis/     # 508 visualisation overlays
                └── prompts_vis/         # Visual prompts (no/, low/, high/)
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hurricane_debris.config import CATEGORY_QUERIES, DataConfig
from hurricane_debris.data.transforms import (
    get_train_transforms,
    get_train_spatial_transforms,
    get_val_transforms,
    get_val_spatial_transforms,
    normalize_and_tensorize,
    stack_instance_masks,
)
from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.designsafe")

# DesignSafe debris mask values → unified taxonomy
# 0 = no debris → 0 (background)
# 1 = low-density debris → 3 (building_damaged — general debris)
# 2 = high-density debris → 3 (building_damaged — general debris)
_DEBRIS_CLASS_MAP = {
    0: 0,  # no debris → background
    1: 3,  # low-density debris → building_damaged
    2: 3,  # high-density debris → building_damaged
}

# Paths within the PRJ-6029 project (tried in order)
_DATA_V2 = "PRJ-6029/Project--hurricane-induced-debris-segmentation-dataset-using-aerial-imagery--V2/data"
_DATA_V1 = "PRJ-6029/Project--hurricane-induced-debris-segmentation-dataset-using-aerial-imagery/data"


class DesignSafeDataset(Dataset):
    """
    PyTorch Dataset for DesignSafe-CI hurricane debris segmentation.

    Like RescueNet, this dataset uses paired image/mask files:
      - ``original/`` contains RGB aerial images
      - ``annotations/`` contains grayscale segmentation masks

    Images without a corresponding mask in ``annotations/`` are treated
    as debris-free (background only).  Connected-component analysis
    extracts bounding boxes from the masks for detection tasks.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        config: Optional[DataConfig] = None,
        task: str = "combined",
        min_component_area: int = 50,
    ):
        self.root_dir = Path(root_dir)
        self.config = config or DataConfig()
        self.split = split
        self.task = task
        self.image_size = self.config.image_size
        self.min_area = min_component_area

        # ── Resolve data directories ────────────────────────────────────
        self.data_dir = self._resolve_data_dir()
        self.img_dir = self.data_dir / "original"
        self.mask_dir = self.data_dir / "annotations"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}. "
                "Please download the DesignSafe-CI dataset (PRJ-6029)."
            )

        # ── Discover samples ────────────────────────────────────────────
        all_samples = self._discover_samples()

        # ── Split (no official splits; deterministic random split) ───────
        self.samples = self._apply_split(all_samples, split)

        logger.info(
            "DesignSafe [%s]: %d samples (%d with masks) from %s",
            split,
            len(self.samples),
            sum(1 for _, m in self.samples if m is not None),
            self.img_dir,
        )

        # ── Transforms (split spatial + normalize for raw_image support) ─
        if split == "train" and self.config.augment_train:
            self.spatial_transform = get_train_spatial_transforms(
                image_size=self.image_size,
                crop_scale=self.config.random_crop_scale,
            )
        else:
            self.spatial_transform = get_val_spatial_transforms(
                image_size=self.image_size,
            )
        self._norm_mean = self.config.image_mean
        self._norm_std = self.config.image_std

    # ── Discovery ────────────────────────────────────────────────────────

    def _resolve_data_dir(self) -> Path:
        """Locate the V2 (preferred) or V1 data directory."""
        for subpath in (_DATA_V2, _DATA_V1):
            candidate = self.root_dir / subpath
            if candidate.exists():
                return candidate
        # Maybe root_dir already points to the data directory
        if (self.root_dir / "original").exists():
            return self.root_dir
        raise FileNotFoundError(
            f"Cannot find DesignSafe data directory under {self.root_dir}. "
            "Expected PRJ-6029/Project--hurricane-induced-debris-segmentation-"
            "dataset-using-aerial-imagery--V2/data/original/"
        )

    def _discover_samples(self) -> List[Tuple[Path, Optional[Path]]]:
        """Find all images and pair them with masks where available."""
        img_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        mask_stems = set()
        if self.mask_dir.exists():
            mask_stems = {
                p.stem for p in self.mask_dir.iterdir()
                if p.suffix.lower() in img_extensions
            }

        samples = []
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in img_extensions:
                continue
            mask_path = None
            if img_path.stem in mask_stems:
                mask_path = self.mask_dir / img_path.name
                if not mask_path.exists():
                    # Try matching just the stem with .png
                    mask_path = self.mask_dir / f"{img_path.stem}.png"
                if not mask_path.exists():
                    mask_path = None
            samples.append((img_path, mask_path))

        return samples

    def _apply_split(
        self, samples: List[Tuple[Path, Optional[Path]]], split: str,
    ) -> List[Tuple[Path, Optional[Path]]]:
        """Deterministic 70/15/15 split, stratified by has-mask."""
        with_mask = [(p, m) for p, m in samples if m is not None]
        without_mask = [(p, m) for p, m in samples if m is None]

        rng = random.Random(self.config.split_seed)
        rng.shuffle(with_mask)
        rng.shuffle(without_mask)

        def _split_list(lst):
            n = len(lst)
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)
            return {
                "train": lst[:n_train],
                "val": lst[n_train:n_train + n_val],
                "test": lst[n_train + n_val:],
            }

        mask_splits = _split_list(with_mask)
        nomask_splits = _split_list(without_mask)

        return mask_splits[split] + nomask_splits[split]

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Cannot read image: %s", img_path)
            return self._blank(idx)
        # Handle RGBA images (DesignSafe originals are RGBA)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Load mask ───────────────────────────────────────────────────
        h_img, w_img = image.shape[:2]
        if mask_path is not None and mask_path.exists():
            semantic_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if semantic_mask is None:
                semantic_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            else:
                # Remap to unified taxonomy
                remapped = np.zeros_like(semantic_mask)
                for src, dst in _DEBRIS_CLASS_MAP.items():
                    remapped[semantic_mask == src] = dst
                semantic_mask = remapped
                # Resize to match image if needed
                h_m, w_m = semantic_mask.shape[:2]
                if (h_m, w_m) != (h_img, w_img):
                    semantic_mask = cv2.resize(
                        semantic_mask, (w_img, h_img),
                        interpolation=cv2.INTER_NEAREST,
                    )
        else:
            semantic_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # ── Extract instances via connected components ──────────────────
        bboxes, category_ids, instance_masks = self._mask_to_instances(
            semantic_mask, w_img, h_img
        )

        # ── Augmentation ────────────────────────────────────────────────
        try:
            if bboxes:
                transformed = self.spatial_transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids,
                    masks=instance_masks,
                )
                aug_image = transformed["image"]
                bboxes = list(transformed["bboxes"])
                category_ids = list(transformed["category_ids"])
                instance_masks = transformed.get("masks", [])
            else:
                transformed = self.spatial_transform(
                    image=image, bboxes=[], category_ids=[],
                )
                aug_image = transformed["image"]
        except Exception as e:
            logger.warning("Transform failed for %s: %s", img_path.name, e)
            return self._blank(idx)

        from PIL import Image as _PILImage
        raw_image = _PILImage.fromarray(aug_image)
        image_t = normalize_and_tensorize(aug_image, self._norm_mean, self._norm_std)

        # Resize semantic mask for evaluation
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
            "masks": stack_instance_masks(instance_masks, self.image_size),
            "semantic_mask": torch.from_numpy(sem_mask_resized).long(),
        }

        return {
            "pixel_values": image_t,
            "raw_image": raw_image,
            "target": target,
            "image_id": idx,
            "image_path": str(img_path),
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _mask_to_instances(
        self, semantic_mask: np.ndarray, img_w: int, img_h: int,
    ) -> Tuple[List, List, List]:
        """Extract bounding boxes and instance masks via connected components."""
        bboxes, cat_ids, inst_masks = [], [], []

        for class_id in sorted(set(_DEBRIS_CLASS_MAP.values()) - {0}):
            binary = (semantic_mask == class_id).astype(np.uint8)
            if binary.sum() == 0:
                continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8,
            )
            for label_idx in range(1, num_labels):
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

    def has_foreground(self, idx: int) -> bool:
        """Lightweight check for non-background content (for empty-sample filtering)."""
        _, mask_path = self.samples[idx]
        if mask_path is None:
            return False
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        return bool(np.any(mask > 0))

    def _blank(self, idx: int) -> Dict:
        return {
            "pixel_values": torch.zeros(3, self.image_size, self.image_size),
            "target": {
                "bboxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": [],
                "category_ids": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros(
                    (0, self.image_size, self.image_size), dtype=torch.float32,
                ),
                "semantic_mask": torch.zeros(
                    self.image_size, self.image_size, dtype=torch.long,
                ),
            },
            "image_id": idx,
            "image_path": "",
        }
