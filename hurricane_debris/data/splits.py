"""
Train / validation / test splitting utilities.
Supports stratified splitting by category distribution and per-dataset official splits.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.splits")


def create_splits(
    annotation_file: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by_category: bool = True,
) -> Dict[str, List[int]]:
    """
    Split image IDs into train / val / test sets.

    Args:
        annotation_file: Path to COCO-format annotation JSON.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.
        stratify_by_category: If True, stratify by dominant category per image.

    Returns:
        Dict with keys "train", "val", "test" mapping to lists of image IDs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    with open(annotation_file) as f:
        data = json.load(f)

    image_ids = [img["id"] for img in data["images"]]

    if not stratify_by_category or "annotations" not in data:
        return _random_split(image_ids, train_ratio, val_ratio, seed)

    # Build image → dominant category mapping for stratification
    img_to_cats: Dict[int, List[int]] = defaultdict(list)
    for ann in data["annotations"]:
        img_to_cats[ann["image_id"]].append(ann["category_id"])

    # Dominant category = most frequent category in that image
    img_to_dominant: Dict[int, int] = {}
    for img_id in image_ids:
        cats = img_to_cats.get(img_id, [0])
        img_to_dominant[img_id] = max(set(cats), key=cats.count)

    # Group by dominant category
    cat_to_imgs: Dict[int, List[int]] = defaultdict(list)
    for img_id, cat in img_to_dominant.items():
        cat_to_imgs[cat].append(img_id)

    train_ids, val_ids, test_ids = [], [], []
    rng = random.Random(seed)

    for cat, ids in cat_to_imgs.items():
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, math.floor(n * train_ratio))
        n_val = max(1, math.floor(n * val_ratio)) if n > 2 else 0
        # Rest goes to test
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    logger.info(
        "Split created — train: %d, val: %d, test: %d", len(train_ids), len(val_ids), len(test_ids)
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _random_split(
    image_ids: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    """Non-stratified random split."""
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def load_official_split(split_dir: str, split: str) -> Optional[List[str]]:
    """
    Load an official split file (one filename per line).

    Args:
        split_dir: Directory containing train.txt / val.txt / test.txt
        split: One of "train", "val", "test".

    Returns:
        List of filenames, or None if file doesn't exist.
    """
    split_file = Path(split_dir) / f"{split}.txt"
    if not split_file.exists():
        logger.warning("Official split file not found: %s", split_file)
        return None
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]
