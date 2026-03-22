"""
Centralized configuration via dataclasses.
All hyperparameters, paths, and model settings in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ── Debris taxonomy (shared across all datasets) ────────────────────────────

DEBRIS_CATEGORIES = {
    0: {"name": "background", "supercategory": "none"},
    1: {"name": "water", "supercategory": "water"},
    2: {"name": "building_no_damage", "supercategory": "structural"},
    3: {"name": "building_damaged", "supercategory": "structural"},
    4: {"name": "vegetation", "supercategory": "vegetation"},
    5: {"name": "road_no_damage", "supercategory": "infrastructure"},
    6: {"name": "road_damaged", "supercategory": "infrastructure"},
    7: {"name": "vehicle", "supercategory": "vehicle"},
}

# Natural language queries for Florence-2 open-vocabulary detection
CATEGORY_QUERIES = {
    1: "flooded area with standing water",
    2: "intact undamaged building",
    3: "damaged or collapsed building with debris",
    4: "vegetation and downed trees",
    5: "intact undamaged road",
    6: "damaged road with cracks or debris",
    7: "vehicle or vehicle wreckage",
}

# RescueNet class mapping → unified taxonomy
RESCUENET_CLASS_MAP = {
    0: 0,  # Background → background
    1: 1,  # Water → water
    2: 2,  # Building-No-Damage → building_no_damage
    3: 3,  # Building-Damaged → building_damaged
    4: 4,  # Vegetation → vegetation
    5: 5,  # Road-No-Damage → road_no_damage
    6: 6,  # Road-Damaged → road_damaged
    7: 7,  # Vehicle → vehicle
}

# Official RescueNet 11-class release (including the ColorMasks-RescueNet
# archive) → project-wide unified 8-class taxonomy
RESCUENET_OFFICIAL_CLASS_MAP = {
    0: 0,   # Background → background
    1: 1,   # Water → water
    2: 2,   # Building-No-Damage → building_no_damage
    3: 3,   # Building-Minor-Damage → building_damaged
    4: 3,   # Building-Major-Damage → building_damaged
    5: 3,   # Building-Total-Destruction → building_damaged
    6: 7,   # Vehicle → vehicle
    7: 5,   # Road-Clear → road_no_damage
    8: 6,   # Road-Blocked → road_damaged
    9: 4,   # Tree → vegetation
    10: 1,  # Pool → water
}

# MSNet/ISBDA damage level mapping → unified taxonomy
# Per Zhu et al. WACV 2021: 1=Slight, 2=Severe, 3=Debris (all are damage)
MSNET_DAMAGE_MAP = {
    0: 0,  # No damage → background
    1: 3,  # Slight damage → building_damaged
    2: 3,  # Severe damage → building_damaged
    3: 3,  # Debris (collapsed) → building_damaged
}


# ── Data configuration ──────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Dataset and data-loading configuration."""

    dataset_root: str = "./datasets"

    # Image settings
    image_size: int = 512
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Split ratios (only used when dataset lacks official splits)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # DataLoader
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    augment_train: bool = True
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    color_jitter_p: float = 0.3
    gauss_noise_p: float = 0.2

    # Number of debris/damage categories (excluding background)
    num_classes: int = 7

    @property
    def root_path(self) -> Path:
        return Path(self.dataset_root)


# ── Florence-2 configuration ────────────────────────────────────────────────

@dataclass
class Florence2Config:
    """Florence-2 model and LoRA training configuration."""

    model_id: str = "microsoft/Florence-2-base-ft"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    max_new_tokens: int = 512
    num_beams: int = 3

    # Early stopping
    early_stopping_patience: int = 3

    # Output
    output_dir: str = "./models/florence2_debris"


# ── SAM2 configuration ──────────────────────────────────────────────────────

@dataclass
class SAM2Config:
    """SAM2 model and fine-tuning configuration."""

    checkpoint_path: str = "./checkpoints/sam2_hiera_large.pt"
    model_cfg: str = "sam2_hiera_l.yaml"

    # Fine-tuning strategy
    freeze_image_encoder: bool = True
    train_prompt_encoder: bool = True
    train_mask_decoder: bool = True

    # Training
    num_epochs: int = 20
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    multimask_output: bool = True

    # Loss weights
    bce_weight: float = 1.0
    dice_weight: float = 1.0

    # Early stopping
    early_stopping_patience: int = 5

    # Output
    output_dir: str = "./models/sam2_debris"


# ── Evaluation configuration ────────────────────────────────────────────────

@dataclass
class EvalConfig:
    """Evaluation metric configuration."""

    iou_thresholds: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    f1_threshold: float = 0.5
    num_classes: int = 7  # excluding background

    # Cross-dataset evaluation
    train_dataset: str = "rescuenet"
    test_datasets: List[str] = field(
        default_factory=lambda: ["designsafe", "msnet"]
    )


# ── Top-level experiment configuration ──────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Full experiment configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    florence2: Florence2Config = field(default_factory=Florence2Config)
    sam2: SAM2Config = field(default_factory=SAM2Config)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # General
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    seed: int = 42
    log_file: Optional[str] = "./logs/training.log"

    def resolve_device(self) -> str:
        """Resolve 'auto' to the best available device."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
