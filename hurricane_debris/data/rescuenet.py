"""
RescueNet Dataset Loader
========================
Alam et al., IEEE TGRS 2022.
4,494 images at 0.5–2 cm/px GSD from Hurricane Michael (FL, 2018).
8-class pixel-level semantic segmentation with official train/val/test splits.

Supported directory layouts (after download):
    datasets/rescuenet/
    ├── train/
    │   ├── train-org-img/      # RGB images
    │   └── train-label-img/    # Semantic masks (class-ID or official colour mask)
    ├── val/
    │   ├── val-org-img/
    │   └── val-label-img/
    └── test/
        ├── test-org-img/
        └── test-label-img/

or the Dropbox layout where imagery and colour masks are extracted as sibling
directories:
    datasets/
    ├── RescueNet/
    │   └── {train,val,test}/{...}-org-img/
    └── ColorMasks-RescueNet/
        └── {train,val,test}/{...}-label-img/
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from hurricane_debris.config import (
    CATEGORY_QUERIES,
    DEBRIS_CATEGORIES,
    RESCUENET_CLASS_MAP,
    RESCUENET_OFFICIAL_CLASS_MAP,
    DataConfig,
)
from hurricane_debris.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    stack_instance_masks,
)
from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.rescuenet")

# The folder naming convention inside each split
_SPLIT_DIRS = {
    "train": ("train/train-org-img", "train/train-label-img"),
    "val": ("val/val-org-img", "val/val-label-img"),
    "test": ("test/test-org-img", "test/test-label-img"),
}

# Official RescueNet RGB palette from the dataset release. OpenCV loads colour
# images in BGR order, so masks are converted back to RGB before lookup.
_COLOUR_MASK_CLASS_IDS = OrderedDict([
    ((0, 0, 0), 0),        # Background
    ((61, 230, 250), 1),   # Water
    ((180, 120, 120), 2),  # Building-No-Damage
    ((235, 255, 7), 3),    # Building-Minor-Damage
    ((255, 184, 6), 4),    # Building-Major-Damage
    ((255, 0, 0), 5),      # Building-Total-Destruction
    ((255, 0, 245), 6),    # Vehicle
    ((140, 140, 140), 7),  # Road-Clear
    ((160, 150, 20), 8),   # Road-Blocked
    ((4, 250, 7), 9),      # Tree
    ((255, 235, 0), 10),   # Pool
])
_COLOUR_MASK_ROOT_NAMES = (
    "ColorMasks-RescueNet",
    "ColourMasks-RescueNet",
    "colormasks-rescuenet",
    "colourmasks-rescuenet",
    "colormask-rescuenet",
    "colourmask-rescuenet",
)


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
            root_dir: Path to the RescueNet dataset root, or to a parent
                      directory containing sibling ``RescueNet/`` and
                      ``ColorMasks-RescueNet/`` folders from Dropbox.
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

        self.img_dir, self.mask_dir = self._resolve_split_dirs()

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}. "
                "Please download RescueNet and extract it following one of the supported layouts."
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
            mask_path = self._resolve_mask_path(img_path)
            if mask_path.exists():
                samples.append((img_path, mask_path))
            else:
                logger.warning("No mask found for %s", img_path.name)
        return samples

    def _resolve_mask_path(self, img_path: Path) -> Path:
        """Resolve the matching RescueNet mask path for an image file."""
        candidates = [
            self.mask_dir / img_path.name,
            self.mask_dir / f"{img_path.stem}.png",
            self.mask_dir / f"{img_path.stem}_lab.png",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[-1]

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

        # Keep a raw PIL copy for Florence-2 processor (avoids denormalize round-trip)
        raw_pil = Image.fromarray(
            cv2.resize(image, (self.image_size, self.image_size))
        )

        # Load semantic mask (pixel value = class ID or official RGB colour mask)
        semantic_mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if semantic_mask_raw is None:
            logger.warning("Cannot read mask: %s", mask_path)
            return self._blank_sample(idx)
        semantic_mask, use_official_map = self._decode_semantic_mask(
            semantic_mask_raw, mask_path
        )

        # Remap classes through the unified taxonomy
        semantic_mask = self._remap_classes(
            semantic_mask,
            official_release=use_official_map,
        )

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
            "masks": stack_instance_masks(instance_masks, self.image_size),
            "semantic_mask": torch.from_numpy(sem_mask_resized).long(),
        }

        return {
            "pixel_values": image_t,
            "raw_image": raw_pil,
            "target": target,
            "image_id": idx,
            "image_path": str(img_path),
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _remap_classes(mask: np.ndarray, official_release: bool = False) -> np.ndarray:
        """Remap RescueNet class IDs through the unified taxonomy."""
        out = np.zeros_like(mask)
        class_map = RESCUENET_OFFICIAL_CLASS_MAP if official_release else RESCUENET_CLASS_MAP
        for src, dst in class_map.items():
            out[mask == src] = dst
        return out

    def _resolve_split_dirs(self) -> Tuple[Path, Path]:
        """Resolve image and mask directories for both flat and Dropbox layouts."""
        img_subdir, mask_subdir = _SPLIT_DIRS[self.split]

        root_candidates = [self.root_dir, self.root_dir / "RescueNet"]

        img_dir = self._first_existing_dir(root_candidates, img_subdir)
        if img_dir is None:
            return self.root_dir / img_subdir, self.root_dir / mask_subdir

        mask_candidates = [*root_candidates, self.root_dir.parent]
        for base in tuple(mask_candidates):
            for name in _COLOUR_MASK_ROOT_NAMES:
                candidate = base / name
                if candidate not in mask_candidates:
                    mask_candidates.append(candidate)

        mask_dir = self._first_existing_dir(mask_candidates, mask_subdir)
        if mask_dir is None:
            mask_dir = img_dir.parent / Path(mask_subdir).name

        return img_dir, mask_dir

    @staticmethod
    def _first_existing_dir(roots: List[Path], subdir: str) -> Optional[Path]:
        for root in roots:
            candidate = root / subdir
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _decode_semantic_mask(mask: np.ndarray, mask_path: Path) -> Tuple[np.ndarray, bool]:
        """Decode grayscale class-ID masks and official RescueNet colour masks."""
        if mask.ndim == 2:
            return mask, False

        if mask.ndim == 3 and mask.shape[2] == 1:
            return mask[:, :, 0], False

        if mask.ndim != 3 or mask.shape[2] < 3:
            logger.warning(
                "Unexpected RescueNet mask shape for %s: %s",
                mask_path,
                getattr(mask, "shape", None),
            )
            return np.zeros(mask.shape[:2], dtype=np.uint8), False

        rgb_mask = mask[:, :, 2::-1]
        if np.array_equal(rgb_mask[:, :, 0], rgb_mask[:, :, 1]) and np.array_equal(
            rgb_mask[:, :, 1], rgb_mask[:, :, 2]
        ):
            return rgb_mask[:, :, 0], False

        flat_rgb = rgb_mask.reshape(-1, 3)
        unique_colors, inverse = np.unique(flat_rgb, axis=0, return_inverse=True)
        color_class_ids = np.zeros(len(unique_colors), dtype=np.uint8)
        unknown_colors = []

        for idx, color in enumerate(unique_colors):
            color_tuple = tuple(int(v) for v in color)
            class_id = _COLOUR_MASK_CLASS_IDS.get(color_tuple)
            if class_id is None:
                unknown_colors.append(color_tuple)
                continue
            color_class_ids[idx] = class_id

        if unknown_colors:
            logger.warning(
                "Found %d unknown colours in RescueNet mask %s; treating them as background. "
                "Examples: %s",
                len(unknown_colors),
                mask_path,
                unknown_colors[:5],
            )

        decoded = color_class_ids[inverse].reshape(rgb_mask.shape[:2])
        return decoded, True

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

    def has_foreground(self, idx: int) -> bool:
        """Lightweight check: does sample *idx* have any non-background mask pixels?

        Loads only the mask (no image, no transforms, no connected-component
        analysis) so it runs ~10x faster than a full ``__getitem__`` call.
        """
        _, mask_path = self.samples[idx]
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            return False
        semantic_mask, use_official = self._decode_semantic_mask(mask_raw, mask_path)
        semantic_mask = self._remap_classes(semantic_mask, official_release=use_official)
        return bool(np.any(semantic_mask > 0))

    def _blank_sample(self, idx: int) -> Dict:
        return {
            "pixel_values": torch.zeros(3, self.image_size, self.image_size),
            "raw_image": Image.new("RGB", (self.image_size, self.image_size)),
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
            "image_path": "",
        }
