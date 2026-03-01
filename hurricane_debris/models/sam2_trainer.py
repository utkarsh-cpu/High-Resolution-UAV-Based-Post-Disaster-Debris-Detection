"""
SAM2 Fine-Tuning for Debris Segmentation (FIXED)
=================================================
Key fixes vs. first_draft.py:
  1. Forward pass goes through model sub-modules DIRECTLY (image_encoder →
     prompt_encoder → mask_decoder) instead of calling predictor.predict()
     which internally uses torch.no_grad() and breaks the autograd graph.
  2. Added proper validation loop.
  3. Checkpointing on best *validation* metric, not training loss.
  4. Integrated with centralized config and logging.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hurricane_debris.config import SAM2Config
from hurricane_debris.utils.logging import get_logger

logger = get_logger("models.sam2")


class SAM2Trainer:
    """
    Fine-tune SAM2 mask decoder + prompt encoder on aerial disaster imagery.
    Image encoder is frozen to preserve general visual features.
    """

    def __init__(
        self,
        config: Optional[SAM2Config] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or SAM2Config()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self._load_model()

    # ── Model loading ────────────────────────────────────────────────────

    def _load_model(self):
        logger.info("Loading SAM2 model from %s", self.cfg.checkpoint_path)
        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install from: "
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        self.model = build_sam2(self.cfg.model_cfg, self.cfg.checkpoint_path)
        self.model = self.model.to(self.device)

    def setup_fine_tuning(self):
        """Freeze image encoder; unfreeze prompt encoder + mask decoder."""
        if self.cfg.freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        if self.cfg.train_prompt_encoder:
            for param in self.model.sam_prompt_encoder.parameters():
                param.requires_grad = True

        if self.cfg.train_mask_decoder:
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = True

        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Trainable parameters: %s / %s (%.2f%%)",
            f"{trainable:,}", f"{total:,}", 100 * trainable / total,
        )

    # ── Fixed forward pass ───────────────────────────────────────────────

    def _forward_sam(
        self,
        image: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAM2 sub-modules, keeping the autograd graph intact.

        FIX: Instead of calling predictor.predict() (which uses no_grad),
        we directly call:
          1. image_encoder  (frozen, but still part of the graph for the decoder)
          2. sam_prompt_encoder  (trainable)
          3. sam_mask_decoder    (trainable)

        Args:
            image: [3, H, W] normalised image tensor on device.
            bboxes: [N, 4] COCO-format boxes [x, y, w, h] on device.

        Returns:
            masks_pred: [N, H, W] predicted mask logits.
            iou_scores: [N] predicted IoU scores.
        """
        # ── 1. Image embedding (frozen encoder) ─────────────────────────
        # SAM2 expects [B, C, H, W] input
        img_input = image.unsqueeze(0)  # [1, 3, H, W]

        with torch.no_grad():
            image_embedding = self.model.image_encoder(img_input)
        # image_embedding may be a dict or tuple depending on SAM2 version
        if isinstance(image_embedding, dict):
            backbone_out = image_embedding
        else:
            backbone_out = {"vision_features": image_embedding}

        # Get the feature map for the mask decoder
        # SAM2's _prepare_backbone_features handles the conversion
        if hasattr(self.model, '_prepare_backbone_features'):
            _, vision_feats, vision_pos_embeds, feat_sizes = (
                self.model._prepare_backbone_features(backbone_out)
            )
            # Get high-res feature maps for mask prediction
            # vision_feats[-1] is the highest resolution feature
            B = 1
            # Flatten vision features to expected SAM2 format
            image_embed = vision_feats[-1].reshape(B, -1, feat_sizes[-1][0], feat_sizes[-1][1])
            high_res_feats = [
                vf.reshape(B, -1, fs[0], fs[1])
                for vf, fs in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            image_embed = backbone_out if torch.is_tensor(backbone_out) else backbone_out["vision_features"]
            high_res_feats = []

        all_masks = []
        all_ious = []

        for bbox in bboxes:
            # ── 2. Convert COCO bbox to SAM2 prompt format ──────────────
            x, y, w, h = bbox
            box_coords = torch.tensor(
                [[x, y, x + w, y + h]], dtype=torch.float32, device=self.device
            )  # [1, 4]  (xyxy format for SAM)

            # ── 3. Prompt encoder (trainable) ───────────────────────────
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=None,
                boxes=box_coords,
                masks=None,
            )

            # ── 4. Mask decoder (trainable) ─────────────────────────────
            low_res_masks, iou_predictions = self.model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.cfg.multimask_output,
                high_res_features=high_res_feats if high_res_feats else None,
            )

            # Select best mask by predicted IoU
            best_idx = iou_predictions.argmax(dim=-1)  # [1]
            best_mask = low_res_masks[0, best_idx]  # [1, H_low, W_low]

            # Upscale to input resolution
            h_img, w_img = image.shape[1], image.shape[2]
            mask_upscaled = F.interpolate(
                best_mask.unsqueeze(0),
                size=(h_img, w_img),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # [H, W]

            all_masks.append(mask_upscaled)
            all_ious.append(iou_predictions[0, best_idx])

        if all_masks:
            return torch.stack(all_masks), torch.stack(all_ious)
        return torch.zeros(0, image.shape[1], image.shape[2], device=self.device), \
               torch.zeros(0, device=self.device)

    # ── Loss computation ─────────────────────────────────────────────────

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Combined Dice + BCE loss for segmentation."""
        # Binary cross-entropy with logits
        bce = F.binary_cross_entropy_with_logits(
            pred.float(), target.float()
        )

        # Dice loss (applied to sigmoid of predictions)
        pred_sigmoid = torch.sigmoid(pred.float())
        target_f = target.float()
        intersection = (pred_sigmoid * target_f).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            pred_sigmoid.sum() + target_f.sum() + 1.0
        )

        return self.cfg.bce_weight * bce + self.cfg.dice_weight * dice

    # ── Training loop ────────────────────────────────────────────────────

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """Train for one epoch using direct forward pass (fixed autograd)."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch["pixel_values"].to(self.device)
            targets = batch["target"]

            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for i in range(images.shape[0]):
                bboxes = targets["bboxes"][i].to(self.device)
                masks_gt = targets["masks"][i].to(self.device)

                if len(bboxes) == 0 or len(masks_gt) == 0:
                    continue

                # FIXED: Direct forward through sub-modules, graph preserved
                masks_pred, _ = self._forward_sam(images[i], bboxes)

                if len(masks_pred) == 0:
                    continue

                # Match predictions to ground truth
                n = min(len(masks_pred), len(masks_gt))
                for j in range(n):
                    loss = self.compute_loss(masks_pred[j], masks_gt[j])
                    batch_loss = batch_loss + loss

            if batch_loss.requires_grad and batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                total_loss += batch_loss.item()

            n_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            images = batch["pixel_values"].to(self.device)
            targets = batch["target"]

            for i in range(images.shape[0]):
                bboxes = targets["bboxes"][i].to(self.device)
                masks_gt = targets["masks"][i].to(self.device)

                if len(bboxes) == 0 or len(masks_gt) == 0:
                    continue

                masks_pred, _ = self._forward_sam(images[i], bboxes)
                n = min(len(masks_pred), len(masks_gt))
                for j in range(n):
                    total_loss += self.compute_loss(masks_pred[j], masks_gt[j]).item()

            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── Full training ────────────────────────────────────────────────────

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: Optional[str] = None,
    ):
        """Complete SAM2 fine-tuning loop with validation and checkpointing."""
        output_dir = output_dir or self.cfg.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting SAM2 training for %d epochs", self.cfg.num_epochs
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self._collate_fn,
                pin_memory=True,
            )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.num_epochs
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.cfg.num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            scheduler.step()

            log_msg = f"Epoch {epoch + 1}/{self.cfg.num_epochs} — train_loss: {train_loss:.4f}"

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                log_msg += f", val_loss: {val_loss:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(output_dir, "best_model.pth"),
                    )
                    logger.info("Saved best model (val_loss: %.4f)", val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d)",
                            epoch + 1,
                            self.cfg.early_stopping_patience,
                        )
                        break
            else:
                # No validation: checkpoint on train loss
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(output_dir, "best_model.pth"),
                    )

            logger.info(log_msg)

        # Save final checkpoint
        torch.save(
            self.model.state_dict(),
            os.path.join(output_dir, "final_model.pth"),
        )
        logger.info("SAM2 training complete. Models saved to %s", output_dir)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "target": {
                "bboxes": [b["target"]["bboxes"] for b in batch],
                "masks": [b["target"]["masks"] for b in batch],
            },
        }
