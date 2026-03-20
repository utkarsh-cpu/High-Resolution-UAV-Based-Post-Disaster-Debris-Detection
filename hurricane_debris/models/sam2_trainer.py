"""
SAM2 Fine-Tuning for Debris Segmentation (OPTIMIZED)
=====================================================
Key fixes vs. first_draft.py:
  1. Forward pass goes through model sub-modules DIRECTLY (image_encoder →
     prompt_encoder → mask_decoder) instead of calling predictor.predict()
     which internally uses torch.no_grad() and breaks the autograd graph.
  2. Added proper validation loop.
  3. Checkpointing on best *validation* metric, not training loss.
  4. Integrated with centralized config and logging.

Performance optimizations:
  - Batched image encoder forward (one call per batch, not per image).
  - Batched box prompts per image (one decoder call per image, not per box).
  - AMP (autocast + GradScaler) for bfloat16/fp16 mixed precision.
  - Gradient accumulation for larger effective batch size.
  - torch.compile on mask decoder for fused kernels.
  - Gradient checkpointing on mask decoder for VRAM savings.
  - Cosine annealing with linear warmup scheduler.
  - DataLoader prefetch_factor for reduced GPU stalls.
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
        self._amp_dtype = None
        self._scaler = None
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
        """Freeze image encoder; unfreeze prompt encoder + mask decoder.
        Also sets up AMP, torch.compile, and gradient checkpointing."""
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

        # ── AMP setup ───────────────────────────────────────────────────
        _is_cuda = getattr(self, 'device', 'cpu') == "cuda"
        if _is_cuda:
            if torch.cuda.is_bf16_supported():
                self._amp_dtype = torch.bfloat16
                # BF16 doesn't need GradScaler
                self._scaler = None
            else:
                self._amp_dtype = torch.float16
                self._scaler = torch.amp.GradScaler("cuda")
            logger.info("AMP enabled with %s", self._amp_dtype)
        else:
            self._amp_dtype = None
            self._scaler = None

        # ── torch.compile on decoder ─────────────────────────────────────
        if _is_cuda:
            try:
                self.model.sam_mask_decoder = torch.compile(
                    self.model.sam_mask_decoder
                )
                logger.info("torch.compile applied to mask decoder")
            except Exception as e:
                logger.warning("torch.compile failed, continuing without: %s", e)

        # ── Gradient checkpointing on decoder ────────────────────────────
        if hasattr(self.model.sam_mask_decoder, 'gradient_checkpointing_enable'):
            self.model.sam_mask_decoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on mask decoder")

        # ── cuDNN benchmark for fixed-size inputs ────────────────────────
        if _is_cuda:
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")

    # ── Batched image encoder ────────────────────────────────────────────

    def _encode_images_batched(
        self, images: torch.Tensor
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Run the frozen image encoder on an entire batch at once.

        Args:
            images: [B, 3, H, W] batch of images on device.

        Returns:
            List of (image_embed, high_res_feats) tuples, one per image.
        """
        B = images.shape[0]

        with torch.no_grad():
            if self._amp_dtype:
                with torch.autocast("cuda", dtype=self._amp_dtype):
                    image_embedding = self.model.image_encoder(images)
            else:
                image_embedding = self.model.image_encoder(images)

        if isinstance(image_embedding, dict):
            backbone_out = image_embedding
        else:
            backbone_out = {"vision_features": image_embedding}

        results = []
        if hasattr(self.model, '_prepare_backbone_features'):
            _, vision_feats, vision_pos_embeds, feat_sizes = (
                self.model._prepare_backbone_features(backbone_out)
            )
            image_embeds = vision_feats[-1].reshape(
                B, -1, feat_sizes[-1][0], feat_sizes[-1][1]
            )
            all_high_res = [
                vf.reshape(B, -1, fs[0], fs[1])
                for vf, fs in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
            for i in range(B):
                hi_res = [hr[i:i+1] for hr in all_high_res]
                results.append((image_embeds[i:i+1], hi_res))
        else:
            raw = backbone_out if torch.is_tensor(backbone_out) else backbone_out["vision_features"]
            for i in range(B):
                results.append((raw[i:i+1], []))

        return results

    # ── Batched forward pass ─────────────────────────────────────────────

    def _forward_sam_batched(
        self,
        image_embed: torch.Tensor,
        high_res_feats: List[torch.Tensor],
        bboxes: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through prompt encoder + mask decoder with ALL boxes
        for a single image in one call.

        Args:
            image_embed: [1, C, H_feat, W_feat] image embedding.
            high_res_feats: list of [1, C, H, W] high-res feature maps.
            bboxes: [N, 4] COCO-format boxes [x, y, w, h] on device.
            img_h: original image height.
            img_w: original image width.

        Returns:
            masks_pred: [N, img_h, img_w] predicted mask logits.
            iou_scores: [N] predicted IoU scores.
        """
        N = bboxes.shape[0]
        if N == 0:
            return (
                torch.zeros(0, img_h, img_w, device=self.device),
                torch.zeros(0, device=self.device),
            )

        # Convert COCO [x, y, w, h] → SAM [x1, y1, x2, y2]
        box_coords = torch.zeros(N, 4, dtype=torch.float32, device=self.device)
        box_coords[:, 0] = bboxes[:, 0]                    # x1
        box_coords[:, 1] = bboxes[:, 1]                    # y1
        box_coords[:, 2] = bboxes[:, 0] + bboxes[:, 2]     # x2
        box_coords[:, 3] = bboxes[:, 1] + bboxes[:, 3]     # y2

        # Prompt encoder — process all boxes at once
        # SAM expects [N, 4] for batched boxes
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None,
            boxes=box_coords,
            masks=None,
        )

        # Expand image embedding to match N prompts
        image_embed_expanded = image_embed.expand(N, -1, -1, -1)
        high_res_expanded = [
            hr.expand(N, -1, -1, -1) for hr in high_res_feats
        ] if high_res_feats else None

        # Mask decoder — all prompts in one call
        low_res_masks, iou_predictions = self.model.sam_mask_decoder(
            image_embeddings=image_embed_expanded,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.cfg.multimask_output,
            high_res_features=high_res_expanded,
        )

        # Select best mask per prompt by predicted IoU
        # low_res_masks: [N, num_masks, H_low, W_low]
        # iou_predictions: [N, num_masks]
        best_idx = iou_predictions.argmax(dim=-1)  # [N]
        best_masks = low_res_masks[
            torch.arange(N, device=self.device), best_idx
        ]  # [N, H_low, W_low]
        best_ious = iou_predictions[
            torch.arange(N, device=self.device), best_idx
        ]  # [N]

        # Upscale to input resolution in one call
        masks_upscaled = F.interpolate(
            best_masks.unsqueeze(1),  # [N, 1, H_low, W_low]
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # [N, H, W]

        return masks_upscaled, best_ious

    # ── Loss computation ─────────────────────────────────────────────────

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Combined Dice + BCE loss for segmentation."""
        bce = F.binary_cross_entropy_with_logits(
            pred.float(), target.float()
        )

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
        """Train for one epoch with AMP, batched encoder, and gradient accumulation."""
        self.model.train()
        # Keep frozen encoder in eval mode
        if self.cfg.freeze_image_encoder:
            self.model.image_encoder.eval()

        total_loss = 0.0
        n_batches = 0
        accum_steps = self.cfg.gradient_accumulation_steps

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            images = batch["pixel_values"].to(self.device, non_blocking=True)
            targets = batch["target"]
            B = images.shape[0]

            # ── Batched image encoding (all images at once) ──────────────
            image_features = self._encode_images_batched(images)

            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for i in range(B):
                bboxes = targets["bboxes"][i].to(self.device, non_blocking=True)
                masks_gt = targets["masks"][i].to(self.device, non_blocking=True)

                if len(bboxes) == 0 or len(masks_gt) == 0:
                    continue

                image_embed, high_res_feats = image_features[i]

                # ── AMP autocast around decoder forward + loss ───────────
                if self._amp_dtype:
                    with torch.autocast("cuda", dtype=self._amp_dtype):
                        masks_pred, _ = self._forward_sam_batched(
                            image_embed, high_res_feats, bboxes,
                            images.shape[2], images.shape[3],
                        )
                else:
                    masks_pred, _ = self._forward_sam_batched(
                        image_embed, high_res_feats, bboxes,
                        images.shape[2], images.shape[3],
                    )

                if len(masks_pred) == 0:
                    continue

                n = min(len(masks_pred), len(masks_gt))
                for j in range(n):
                    loss = self.compute_loss(masks_pred[j], masks_gt[j])
                    batch_loss = batch_loss + loss

            # Scale loss for gradient accumulation
            if batch_loss.requires_grad and batch_loss > 0:
                scaled_loss = batch_loss / accum_steps

                if self._scaler is not None:
                    self._scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Step optimizer every accum_steps
                if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                    if self._scaler is not None:
                        self._scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            max_norm=1.0,
                        )
                        self._scaler.step(optimizer)
                        self._scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            max_norm=1.0,
                        )
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += batch_loss.item()

            n_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Compute validation loss with AMP."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            images = batch["pixel_values"].to(self.device, non_blocking=True)
            targets = batch["target"]

            image_features = self._encode_images_batched(images)

            for i in range(images.shape[0]):
                bboxes = targets["bboxes"][i].to(self.device, non_blocking=True)
                masks_gt = targets["masks"][i].to(self.device, non_blocking=True)

                if len(bboxes) == 0 or len(masks_gt) == 0:
                    continue

                image_embed, high_res_feats = image_features[i]

                if self._amp_dtype:
                    with torch.autocast("cuda", dtype=self._amp_dtype):
                        masks_pred, _ = self._forward_sam_batched(
                            image_embed, high_res_feats, bboxes,
                            images.shape[2], images.shape[3],
                        )
                else:
                    masks_pred, _ = self._forward_sam_batched(
                        image_embed, high_res_feats, bboxes,
                        images.shape[2], images.shape[3],
                    )

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
            prefetch_factor=2,
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
                prefetch_factor=2,
            )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # Cosine annealing with linear warmup
        warmup_epochs = max(1, self.cfg.num_epochs // 10)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.num_epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
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
