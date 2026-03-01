"""
Hurricane Debris Detection - Dataset Preparation & Model Retraining
===================================================================

This script provides:
1. Dataset download and preparation for hurricane debris detection
2. Fine-tuning pipeline for Florence-2 on debris-specific vocabulary
3. Fine-tuning pipeline for SAM2 on aerial disaster imagery
4. Combined training with domain adaptation

Datasets:
- xBD (xView Building Damage) - Building damage assessment
- EIDSeg - Earthquake damage from social media
- Post-Hurricane Debris (custom) - Aerial debris segmentation
- FloodNet - Flood damage and debris detection
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
import requests
import zipfile
import gdown
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================================================
# DATASET DOWNLOADERS
# =============================================================================

class DatasetDownloader:
    """Download and prepare disaster-specific datasets."""
    
    DATASET_URLS = {
        "xbd": {
            "url": "https://challenge.xviewdataset.org/download-links",
            "description": "xView Building Damage - Building damage assessment",
            "size": "~20GB",
            "tasks": ["damage_classification", "building_segmentation"]
        },
        "floodnet": {
            "train": "https://drive.google.com/uc?id=1LnvwnV8XIPHTzP6jH12b8Q6BpugZ2KJE",
            "test": "https://drive.google.com/uc?id=1q7KCnOqtK7vWO4l4ar1tT7Qgu-5q6H0y",
            "description": "FloodNet - Post-flood damage assessment",
            "size": "~2GB",
            "tasks": ["flood_detection", "debris_segmentation"]
        },
        "eidseg": {
            "url": "https://github.com/ai4ce/EID-Seg",
            "description": "Earthquake Image Dataset with Segmentation",
            "size": "~5GB",
            "tasks": ["damage_segmentation", "rubble_detection"]
        },
        "post_hurricane_debris": {
            "url": "https://github.com/Way-Yuhao/CLIPSeg-debris",
            "description": "Post-Hurricane Debris Segmentation Dataset",
            "size": "~1GB",
            "tasks": ["debris_detection", "debris_classification"]
        }
    }
    
    def __init__(self, root_dir: str = "./datasets"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)
    
    def download_floodnet(self) -> Path:
        """
        Download FloodNet dataset for flood damage assessment.
        Best for: Water/flood debris detection
        """
        print("Downloading FloodNet dataset...")
        dataset_dir = self.root_dir / "floodnet"
        dataset_dir.mkdir(exist_ok=True)
        
        # Download train and test sets
        for split, file_id in [("train", self.DATASET_URLS["floodnet"]["train"]),
                               ("test", self.DATASET_URLS["floodnet"]["test"])]:
            output = dataset_dir / f"{split}.zip"
            if not output.exists():
                print(f"Downloading {split} set...")
                gdown.download(file_id, str(output), quiet=False)
                
                # Extract
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir / split)
        
        print(f"✅ FloodNet downloaded to {dataset_dir}")
        return dataset_dir
    
    def download_xbd(self) -> Path:
        """
        Download xBD (xView Building Damage) dataset.
        Best for: Building damage assessment, structural debris
        Requires: Registration at https://challenge.xviewdataset.org
        """
        print("""
        xBD Dataset Download Instructions:
        ==================================
        1. Visit: https://challenge.xviewdataset.org/download-links
        2. Register for an account
        3. Download: train.tar.gz, test.tar.gz, tier3.tar.gz
        4. Extract to: ./datasets/xbd/
        
        Dataset Structure:
        - train/images/ - Pre-disaster images
        - train/images_post/ - Post-disaster images
        - train/labels/ - Building damage labels (0=no damage, 1-4=severity)
        """)
        
        dataset_dir = self.root_dir / "xbd"
        return dataset_dir

    
    def prepare_combined_dataset(self) -> Path:
        """
        Combine multiple datasets into unified format for training.
        """
        print("Preparing combined disaster debris dataset...")
        
        combined_dir = self.root_dir / "combined_disaster"
        (combined_dir / "images").mkdir(parents=True, exist_ok=True)
        (combined_dir / "masks").mkdir(parents=True, exist_ok=True)
        
        # Collect all datasets
        datasets = []
        
        # Add FloodNet if available
        floodnet_dir = self.root_dir / "floodnet"
        if floodnet_dir.exists():
            datasets.append(("floodnet", floodnet_dir))
        
        # Add synthetic data
        syn_dir = self.root_dir / "synthetic_debris"
        if syn_dir.exists():
            datasets.append(("synthetic", syn_dir))
        
        # Merge annotations
        combined_annotations = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "downed_tree", "supercategory": "vegetation"},
                {"id": 2, "name": "building_debris", "supercategory": "structural"},
                {"id": 3, "name": "flooded_area", "supercategory": "water"},
                {"id": 4, "name": "vehicle_debris", "supercategory": "vehicle"},
                {"id": 5, "name": "scattered_rubble", "supercategory": "rubble"},
                {"id": 6, "name": "power_infrastructure", "supercategory": "utility"},
            ]
        }
        
        ann_id = 0
        img_id = 0
        
        for ds_name, ds_dir in datasets:
            ann_file = ds_dir / "annotations" / "instances.json"
            if ann_file.exists():
                with open(ann_file) as f:
                    data = json.load(f)
                
                for img_ann in data.get("images", []):
                    # Copy image
                    src_img = ds_dir / "images" / img_ann["file_name"]
                    dst_img = combined_dir / "images" / f"{ds_name}_{img_ann['file_name']}"
                    
                    if src_img.exists():
                        import shutil
                        shutil.copy(src_img, dst_img)
                        
                        # Add to combined annotations
                        for ann in img_ann.get("annotations", []):
                            combined_annotations["annotations"].append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": ann.get("category_id", 1),
                                "bbox": ann["bbox"],
                                "area": ann["area"],
                                "segmentation": ann.get("segmentation", []),
                                "iscrowd": 0
                            })
                            ann_id += 1
                        
                        combined_annotations["images"].append({
                            "id": img_id,
                            "file_name": f"{ds_name}_{img_ann['file_name']}",
                            "height": img_ann.get("height", 1024),
                            "width": img_ann.get("width", 1024)
                        })
                        img_id += 1
        
        # Save combined annotations
        with open(combined_dir / "annotations.json", 'w') as f:
            json.dump(combined_annotations, f, indent=2)
        
        print(f"✅ Combined dataset: {img_id} images, {ann_id} annotations")
        return combined_dir


# =============================================================================
# PYTORCH DATASETS
# =============================================================================

class HurricaneDebrisDataset(Dataset):
    """
    PyTorch Dataset for hurricane debris detection.
    Supports both detection and segmentation tasks.
    """
    
    DEBRIS_QUERIES = {
        1: "downed tree blocking road",
        2: "collapsed building debris",
        3: "flooded area with water",
        4: "damaged vehicle wreckage",
        5: "scattered rubble and debris",
        6: "fallen power line pole",
    }
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str = "annotations.json",
        split: str = "train",
        image_size: int = 768,
        processor=None,
        sam_processor=None,
        task: str = "detection"  # "detection", "segmentation", or "combined"
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.processor = processor
        self.sam_processor = sam_processor
        self.task = task
        
        # Load annotations
        with open(self.root_dir / annotation_file) as f:
            self.data = json.load(f)
        
        self.images = self.data["images"]
        self.annotations = self.data["annotations"]
        self.categories = {c["id"]: c for c in self.data["categories"]}
        
        # Build image-to-annotations mapping
        self.img_to_ann = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_to_ann:
                self.img_to_ann[img_id] = []
            self.img_to_ann[img_id].append(ann)
        
        # Filter images with annotations
        self.images = [img for img in self.images if img["id"] in self.img_to_ann]
        
        # Augmentations
        if split == "train":
            self.transform = A.Compose([
                A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.root_dir / "images" / img_info["file_name"]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        anns = self.img_to_ann.get(img_info["id"], [])
        
        # Prepare bboxes and labels
        bboxes = []
        category_ids = []
        masks = []
        
        for ann in anns:
            bboxes.append(ann["bbox"])  # [x, y, w, h]
            category_ids.append(ann["category_id"])
            
            # Create mask from segmentation if available
            if "segmentation" in ann and ann["segmentation"]:
                mask = self._polygons_to_mask(
                    ann["segmentation"], 
                    img_info["height"], 
                    img_info["width"]
                )
                masks.append(mask)
        
        # Apply transforms
        if len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids,
                masks=masks if masks else None
            )
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            category_ids = transformed["category_ids"]
            masks = transformed.get("masks", [])
        else:
            # Empty image
            transformed = self.transform(image=image, bboxes=[], category_ids=[])
            image = transformed["image"]
        
        # Prepare Florence-2 format
        text_queries = [self.DEBRIS_QUERIES.get(cat_id, "debris") for cat_id in category_ids]
        
        # Create target for training
        target = {
            "bboxes": torch.tensor(bboxes) if bboxes else torch.zeros((0, 4)),
            "labels": text_queries,
            "category_ids": torch.tensor(category_ids) if category_ids else torch.zeros(0),
            "masks": torch.stack([torch.tensor(m) for m in masks]) if masks else torch.zeros((0, self.image_size, self.image_size))
        }
        
        return {
            "pixel_values": image,
            "target": target,
            "image_id": img_info["id"]
        }
    
    def _polygons_to_mask(self, polygons, height, width):
        """Convert COCO polygon to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            poly = np.array(poly).reshape(-1, 2)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        return mask


# =============================================================================
# FLORENCE-2 FINE-TUNING
# =============================================================================

class Florence2Trainer:
    """
    Fine-tune Florence-2 for hurricane debris detection.
    Uses LoRA for efficient adaptation.
    """
    
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base-ft",
        device: str = None
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = None
        self.model = None
        self._load_base_model()
    
    def _load_base_model(self):
        """Load base Florence-2 model."""
        print("Loading Florence-2 base model...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)
    
    def setup_lora(self, r: int = 16, lora_alpha: int = 32):
        """Setup LoRA for efficient fine-tuning."""
        print(f"Setting up LoRA (r={r}, alpha={lora_alpha})...")
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_debris_data(self, examples):
        """Prepare data batch for Florence-2 training."""
        images = []
        texts = []
        
        for example in examples:
            # Convert tensor to PIL
            img_tensor = example["pixel_values"]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                     np.array([0.485, 0.456, 0.406]))
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(img_np)
            
            # Build text prompt
            target = example["target"]
            queries = target["labels"]
            
            # Create task-specific prompt
            prompt = "<OPEN_VOCABULARY_DETECTION>" + ", ".join(queries)
            
            images.append(image)
            texts.append(prompt)
        
        # Process batch
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Create labels (for causal LM, labels = input_ids)
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        output_dir: str = "./florence2_debris",
        num_epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 5e-5
    ):
        """Fine-tune Florence-2 on debris dataset."""
        print(f"Starting training for {num_epochs} epochs...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            report_to="tensorboard"
        )
        
        # Custom data collator
        def data_collator(examples):
            return self.prepare_debris_data(examples)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else []
        )
        
        trainer.train()
        
        # Save final model
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"✅ Model saved to {output_dir}")
    
    def inference(self, image: Image.Image, query: str) -> Dict:
        """Run inference with fine-tuned model."""
        prompt = f"<OPEN_VOCABULARY_DETECTION>{query}"
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        parsed = self.processor.post_process_generation(
            generated_text,
            task="<OPEN_VOCABULARY_DETECTION>",
            image_size=(image.width, image.height)
        )
        
        return parsed


# =============================================================================
# SAM2 FINE-TUNING
# =============================================================================

class SAM2Trainer:
    """
    Fine-tune SAM2 for hurricane debris segmentation.
    Adapts mask decoder on aerial disaster imagery.
    """
    
    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/sam2_hiera_large.pt",
        model_cfg: str = "sam2_hiera_l.yaml",
        device: str = None
    ):
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.predictor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM2 model."""
        print("Loading SAM2 model...")
        
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2
            
            self.model = build_sam2(self.model_cfg, self.checkpoint_path)
            self.predictor = SAM2ImagePredictor(self.model)
            self.model = self.model.to(self.device)
            
        except ImportError:
            print("SAM2 not installed. Please install from: https://github.com/facebookresearch/segment-anything-2")
            raise
    
    def setup_fine_tuning(self, freeze_encoder: bool = True):
        """Setup model for fine-tuning."""
        if freeze_encoder:
            # Freeze image encoder
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            
            # Only train mask decoder and prompt encoder
            for param in self.model.sam_prompt_encoder.parameters():
                param.requires_grad = True
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def train_epoch(self, dataloader, optimizer, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch["pixel_values"].to(self.device)
            targets = batch["target"]
            
            batch_loss = 0
            
            for i, image in enumerate(images):
                # Get ground truth
                bboxes = targets["bboxes"][i].to(self.device)
                masks_gt = targets["masks"][i].to(self.device)
                
                if len(bboxes) == 0:
                    continue
                
                # Set image
                self.predictor.set_image(image.permute(1, 2, 0).cpu().numpy())
                
                # Forward pass for each object
                for bbox, mask_gt in zip(bboxes, masks_gt):
                    # Prepare box prompt
                    input_box = bbox.cpu().numpy()
                    
                    # Predict mask
                    masks_pred, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=True
                    )
                    
                    # Select best mask
                    best_idx = np.argmax(scores)
                    mask_pred = torch.from_numpy(masks_pred[best_idx]).to(self.device)
                    
                    # Compute loss (dice + ce)
                    loss = self._compute_loss(mask_pred, mask_gt)
                    batch_loss += loss
            
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
            
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}" if batch_loss > 0 else "N/A"})
        
        return total_loss / len(dataloader)
    
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute segmentation loss (Dice + BCE)."""
        # Binary cross entropy
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred.float(), target.float())
        
        # Dice loss
        pred_flat = pred.view(-1).float()
        target_flat = target.view(-1).float()
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2. * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
        
        return bce + dice
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        output_dir: str = "./sam2_debris",
        num_epochs: int = 20,
        batch_size: int = 2,
        learning_rate: float = 1e-5
    ):
        """Fine-tune SAM2 on debris segmentation."""
        print(f"Starting SAM2 training for {num_epochs} epochs...")
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, optimizer, epoch)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.model.state_dict(), f"{output_dir}/best_model.pth")
                print(f"✅ Saved best model (loss: {best_loss:.4f})")
        
        print(f"✅ Training complete. Model saved to {output_dir}")
    
    def _collate_fn(self, batch):
        """Custom collate function."""
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "target": {
                "bboxes": [b["target"]["bboxes"] for b in batch],
                "masks": [b["target"]["masks"] for b in batch]
            }
        }


# =============================================================================
# COMBINED TRAINING PIPELINE
# =============================================================================

class HurricaneDebrisTrainingPipeline:
    """
    End-to-end training pipeline for hurricane debris detection.
    Fine-tunes both Florence-2 and SAM2 with domain adaptation.
    """
    
    def __init__(self, dataset_root: str = "./datasets"):
        self.dataset_root = Path(dataset_root)
        self.downloader = DatasetDownloader(dataset_root)
        
        self.florence_trainer = None
        self.sam2_trainer = None
    
    def prepare_datasets(self):
        """Download and prepare all datasets."""
        print("=" * 60)
        print("STEP 1: Dataset Preparation")
        print("=" * 60)
        
        # Create synthetic data (guaranteed to work)
        syn_dir = self.downloader.create_synthetic_debris_dataset(num_samples=2000)
        
        # Download FloodNet (if available)
        try:
            floodnet_dir = self.downloader.download_floodnet()
        except Exception as e:
            print(f"FloodNet download failed: {e}")
            floodnet_dir = None
        
        # Combine datasets
        combined_dir = self.downloader.prepare_combined_dataset()
        
        return combined_dir
    
    def train_florence2(
        self,
        dataset_dir: str,
        output_dir: str = "./models/florence2_debris",
        epochs: int = 10
    ):
        """Fine-tune Florence-2."""
        print("\n" + "=" * 60)
        print("STEP 2: Fine-tuning Florence-2")
        print("=" * 60)
        
        # Create datasets
        train_dataset = HurricaneDebrisDataset(
            dataset_dir,
            split="train",
            task="detection"
        )
        
        val_dataset = HurricaneDebrisDataset(
            dataset_dir,
            split="val",
            task="detection"
        )
        
        # Initialize trainer
        self.florence_trainer = Florence2Trainer()
        self.florence_trainer.setup_lora(r=16, lora_alpha=32)
        
        # Train
        self.florence_trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=2,
            learning_rate=5e-5
        )
        
        return output_dir
    
    def train_sam2(
        self,
        dataset_dir: str,
        output_dir: str = "./models/sam2_debris",
        epochs: int = 20
    ):
        """Fine-tune SAM2."""
        print("\n" + "=" * 60)
        print("STEP 3: Fine-tuning SAM2")
        print("=" * 60)
        
        # Create datasets
        train_dataset = HurricaneDebrisDataset(
            dataset_dir,
            split="train",
            task="segmentation"
        )
        
        # Initialize trainer
        self.sam2_trainer = SAM2Trainer()
        self.sam2_trainer.setup_fine_tuning(freeze_encoder=True)
        
        # Train
        self.sam2_trainer.train(
            train_dataset=train_dataset,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=2,
            learning_rate=1e-5
        )
        
        return output_dir
    
    def run_full_pipeline(self):
        """Execute complete training pipeline."""
        # Prepare data
        dataset_dir = self.prepare_datasets()
        
        # Train models
        florence_path = self.train_florence2(dataset_dir)
        sam2_path = self.train_sam2(dataset_dir)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Florence-2 model: {florence_path}")
        print(f"SAM2 model: {sam2_path}")
        
        return {
            "florence2": florence_path,
            "sam2": sam2_path,
            "dataset": str(dataset_dir)
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution for training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hurricane Debris Detection Training")
    parser.add_argument("--prepare-data", action="store_true", help="Only prepare datasets")
    parser.add_argument("--train-florence", action="store_true", help="Train Florence-2 only")
    parser.add_argument("--train-sam2", action="store_true", help="Train SAM2 only")
    parser.add_argument("--full-pipeline", action="store_true", help="Run complete pipeline")
    parser.add_argument("--dataset-dir", default="./datasets", help="Dataset root directory")
    parser.add_argument("--epochs-florence", type=int, default=10, help="Florence-2 epochs")
    parser.add_argument("--epochs-sam2", type=int, default=20, help="SAM2 epochs")
    
    args = parser.parse_args()
    
    pipeline = HurricaneDebrisTrainingPipeline(args.dataset_dir)
    
    if args.prepare_data:
        pipeline.prepare_datasets()
    
    elif args.train_florence:
        dataset_dir = pipeline.prepare_datasets()
        pipeline.train_florence2(dataset_dir, epochs=args.epochs_florence)
    
    elif args.train_sam2:
        dataset_dir = pipeline.prepare_datasets()
        pipeline.train_sam2(dataset_dir, epochs=args.epochs_sam2)
    
    elif args.full_pipeline:
        results = pipeline.run_full_pipeline()
        print("\nResults:", json.dumps(results, indent=2))
    
    else:
        print("""
Hurricane Debris Detection - Training Pipeline
==============================================

Usage:
  python train.py --full-pipeline          # Run complete training
  python train.py --prepare-data           # Only download/prepare datasets
  python train.py --train-florence         # Train Florence-2 only
  python train.py --train-sam2             # Train SAM2 only

Options:
  --dataset-dir PATH       Dataset root directory (default: ./datasets)
  --epochs-florence N      Florence-2 training epochs (default: 10)
  --epochs-sam2 N          SAM2 training epochs (default: 20)
        """)


if __name__ == "__main__":
    main()