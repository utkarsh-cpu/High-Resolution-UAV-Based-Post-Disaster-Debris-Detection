"""
Albumentations transform pipelines for training and evaluation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


def get_train_transforms(
    image_size: int = 768,
    crop_scale: Tuple[float, float] = (0.8, 1.0),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    color_jitter_p: float = 0.3,
    gauss_noise_p: float = 0.2,
) -> A.Compose:
    """Return augmentation pipeline for training split."""
    return A.Compose(
        [
            A.RandomResizedCrop(
                height=image_size, width=image_size, scale=crop_scale
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, p=color_jitter_p
            ),
            A.GaussNoise(var_limit=(10, 50), p=gauss_noise_p),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(
    image_size: int = 768,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Return transform pipeline for validation / test splits."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
    )
