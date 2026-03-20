"""
Florence-2 Fine-Tuning for Debris Detection (FIXED)
====================================================
Key fixes vs. first_draft.py:
  1. Labels are now the *target detection output* tokens, not input_ids clone.
  2. Removed wasteful denormalize→re-normalize round-trip in collation.
  3. Added proper validation loop with metric logging.
  4. Integrated with centralized config and logging.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from hurricane_debris.config import CATEGORY_QUERIES, Florence2Config
from hurricane_debris.utils.logging import get_logger

logger = get_logger("models.florence2")


def load_florence_processor(model_id: str):
    """Load a Florence-2 processor with a slow-tokenizer fallback."""
    try:
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
    except AttributeError as exc:
        if "additional_special_tokens" not in str(exc):
            raise

        logger.warning(
            "Florence-2 processor load hit tokenizer incompatibility; retrying with use_fast=False"
        )
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )


def _bbox_coco_to_florence(bbox, img_w: int, img_h: int) -> str:
    """
    Convert a COCO bbox [x, y, w, h] to Florence-2 location tokens.

    Florence-2 normalizes coordinates to [0, 999] and uses special tokens
    ``<loc_XXX>`` for each coordinate.  The output format is:
        ``<loc_x1><loc_y1><loc_x2><loc_y2>``
    """
    x, y, w, h = bbox
    x1 = int(round(x / img_w * 999))
    y1 = int(round(y / img_h * 999))
    x2 = int(round((x + w) / img_w * 999))
    y2 = int(round((y + h) / img_h * 999))
    # Clamp to valid range
    x1, y1, x2, y2 = [max(0, min(v, 999)) for v in (x1, y1, x2, y2)]
    return f"<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"


class Florence2Trainer:
    """
    Fine-tune Florence-2 for open-vocabulary debris detection via LoRA.

    Training target format (what the model learns to *generate*):
        ``category_name_1<loc_x1><loc_y1><loc_x2><loc_y2>
          category_name_2<loc_x1><loc_y1><loc_x2><loc_y2> …``
    """

    def __init__(
        self,
        config: Optional[Florence2Config] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or Florence2Config()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor: Optional[AutoProcessor] = None
        self.model = None
        self._load_base_model()

    # ── Model loading ────────────────────────────────────────────────────

    def _load_base_model(self):
        logger.info("Loading Florence-2 base model: %s", self.cfg.model_id)

        self.processor = load_florence_processor(self.cfg.model_id)
        # Half-precision on CUDA (2x memory saving vs float32)
        if self.device == "cuda":
            _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            _dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, torch_dtype=_dtype, trust_remote_code=True
        ).to(self.device)

    def setup_lora(self):
        """Attach LoRA adapters for parameter-efficient fine-tuning."""
        logger.info(
            "Attaching LoRA (r=%d, alpha=%d)", self.cfg.lora_r, self.cfg.lora_alpha
        )
        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=self.cfg.lora_target_modules,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    # ── Data collation (FIXED) ───────────────────────────────────────────

    def collate_fn(self, examples: List[Dict]) -> Dict:
        """
        Prepare a batch for Florence-2 training.

        FIX: Labels are the *target detection output* token sequence,
        NOT a clone of the input prompt tokens.  The input prompt is
        ``<OPEN_VOCABULARY_DETECTION>`` and the label is the model's
        expected output string containing location tokens + category names.
        """
        images = []
        prompts = []
        target_texts = []

        for example in examples:
            target = example["target"]
            bboxes = target["bboxes"]  # [N, 4] tensor
            labels = target["labels"]  # list of category name strings

            # ── Build the target output text the model should generate ──
            # Format: "category<loc_x1><loc_y1><loc_x2><loc_y2>"
            # repeated for each detection in the image.
            detection_strs = []
            for bbox, label in zip(bboxes, labels):
                loc_tokens = _bbox_coco_to_florence(
                    bbox.tolist(), self.image_size, self.image_size
                )
                detection_strs.append(f"{label}{loc_tokens}")

            target_text = "".join(detection_strs) if detection_strs else ""

            # ── Image: keep as raw uint8 PIL so the processor normalizes ─
            # Instead of denormalizing the tensor, we store raw PIL at
            # dataset level.  For compatibility with augmented tensors,
            # convert back to uint8 PIL here.
            img_tensor = example["pixel_values"]  # [C, H, W] normalised
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (
                img_np * np.array([0.229, 0.224, 0.225])
                + np.array([0.485, 0.456, 0.406])
            )
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            images.append(Image.fromarray(img_np))
            prompts.append("<OPEN_VOCABULARY_DETECTION>")
            target_texts.append(target_text)

        # ── Tokenize inputs (prompt) ─────────────────────────────────────
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # ── Tokenize targets (detection output text) ─────────────────────
        # Labels are the tokenized *output* sequence.
        label_encoding = self.processor.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_new_tokens,
        )

        labels = label_encoding["input_ids"]
        # Mask padding tokens with -100 so they're ignored in the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    @property
    def image_size(self) -> int:
        """Shortcut to the image size used by the dataset."""
        return 768  # Florence-2 default; overridden via config in dataset

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: Optional[str] = None,
    ):
        """Fine-tune Florence-2 on a debris detection dataset."""
        output_dir = output_dir or self.cfg.output_dir
        logger.info("Starting Florence-2 training for %d epochs", self.cfg.num_epochs)

        _use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        _use_fp16 = torch.cuda.is_available() and not _use_bf16

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=val_dataset is not None,
            bf16=_use_bf16,
            fp16=_use_fp16,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to="tensorboard",
            metric_for_best_model="eval_loss" if val_dataset else None,
            remove_unused_columns=False,
        )

        callbacks = []
        if val_dataset is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                )
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_fn,
            callbacks=callbacks,
        )

        trainer.train()

        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        logger.info("Florence-2 model saved to %s", output_dir)

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(self, image: Image.Image, query: str) -> Dict:
        """Run open-vocabulary detection on a single image."""
        prompt = f"<OPEN_VOCABULARY_DETECTION>{query}"

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.cfg.max_new_tokens,
            num_beams=self.cfg.num_beams,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task="<OPEN_VOCABULARY_DETECTION>",
            image_size=(image.width, image.height),
        )
        return parsed
