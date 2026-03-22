"""
Florence-2 Fine-Tuning for Debris Detection (FIXED)
====================================================
Key fixes vs. first_draft.py:
  1. Labels are now the *target detection output* tokens, not input_ids clone.
  2. Removed wasteful denormalize→re-normalize round-trip in collation.
  3. Added proper validation loop with metric logging.
  4. Integrated with centralized config and logging.
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def _preload_hf_datasets_module():
    """Ensure transformers resolves the Hugging Face datasets package.

    The repository has a top-level datasets/ directory that would otherwise
    shadow the external package when Trainer imports its optional integrations.
    """
    existing = sys.modules.get("datasets")
    if existing is not None and hasattr(existing, "Dataset"):
        return

    repo_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("datasets", None)
        sys.path = [
            path for path in sys.path
            if Path(path or ".").resolve() != repo_root
        ]
        hf_datasets = importlib.import_module("datasets")
        sys.modules["datasets"] = hf_datasets
    finally:
        sys.path = original_sys_path


_preload_hf_datasets_module()

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


def _fix_florence2_weight_tying(model):
    """Restore tied embeddings broken by Transformers 5.x.

    Florence-2 shares one embedding tensor across encoder, decoder, and
    lm_head.  The checkpoint stores it under ``language_model.model.shared``
    and expects the library to tie the other three.  Transformers 5.x
    sometimes fails to do this, leaving encoder/decoder/lm_head randomly
    initialized.  This function copies the correct ``shared`` weight to
    all three locations.
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    inner = getattr(lm, "model", None)
    if inner is None:
        return
    shared = getattr(inner, "shared", None)
    if shared is None:
        return

    import torch as _torch

    shared_w = shared.weight.data
    enc_emb = inner.encoder.embed_tokens.weight.data
    dec_emb = inner.decoder.embed_tokens.weight.data
    lm_head = lm.lm_head.weight.data

    needs_fix = not (
        _torch.equal(shared_w, enc_emb)
        and _torch.equal(shared_w, dec_emb)
        and _torch.equal(shared_w, lm_head)
    )
    if needs_fix:
        inner.encoder.embed_tokens.weight = shared.weight
        inner.decoder.embed_tokens.weight = shared.weight
        lm.lm_head.weight = shared.weight
        logger.info(
            "Fixed Florence-2 weight tying: copied shared embedding "
            "(norm=%.1f) to encoder/decoder/lm_head",
            shared_w.norm().item(),
        )


def _ensure_slow_image_processor(processor, model_id: str):
    """Replace fast CLIPImageProcessor with slow version if resize is broken."""
    from transformers import CLIPImageProcessor

    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        if type(ip).__name__ == "CLIPImageProcessorFast":
            processor.image_processor = CLIPImageProcessor.from_pretrained(
                model_id
            )
    return processor


def load_florence_processor(model_id: str):
    """Load a Florence-2 processor with a slow-tokenizer fallback.

    Transformers 5.x may fail to restore ``additional_special_tokens``
    automatically for RobertaTokenizer-based checkpoints.  When that
    happens we load the tokenizer separately, patch the missing tokens
    from the saved ``special_tokens_map.json``, and pass it to the
    processor constructor.

    Also forces the slow CLIPImageProcessor to guarantee correct resizing
    (the fast variant may skip resize in some Transformers versions).
    """
    try:
        proc = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        return _ensure_slow_image_processor(proc, model_id)
    except AttributeError as exc:
        if "additional_special_tokens" not in str(exc):
            raise

        logger.warning(
            "Florence-2 processor load hit tokenizer incompatibility; "
            "patching additional_special_tokens manually"
        )

        from transformers import AutoTokenizer
        import json
        from pathlib import Path

        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

        if not hasattr(tok.__class__, "additional_special_tokens"):
            tok.__class__.additional_special_tokens = property(
                lambda self: list(
                    self._special_tokens_map.get("additional_special_tokens", [])
                )
            )

        stm_path = Path(model_id) / "special_tokens_map.json"
        extra_tokens = []
        if stm_path.exists():
            with open(stm_path) as f:
                stm = json.load(f)
            extra_tokens = [
                t["content"] if isinstance(t, dict) else t
                for t in stm.get("additional_special_tokens", [])
            ]

        # Transformers 5.x stores extra special tokens internally but the
        # Florence remote processor expects a public attribute as well.
        tok._special_tokens_map["additional_special_tokens"] = extra_tokens

        proc = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, tokenizer=tok
        )
        return _ensure_slow_image_processor(proc, model_id)


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
        )
        _fix_florence2_weight_tying(self.model)
        self.model = self.model.to(self.device)

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

        Uses raw PIL images from the dataset directly (no denormalize round-trip).
        Labels are the *target detection output* token sequence.
        """
        images = []
        prompts = []
        target_texts = []

        for example in examples:
            target = example["target"]
            bboxes = target["bboxes"]  # [N, 4] tensor
            labels = target["labels"]  # list of category name strings

            # Use raw PIL image directly — no expensive denormalize round-trip
            if "raw_image" in example:
                pil_img = example["raw_image"]
            else:
                # Fallback: denormalize tensor → PIL (for datasets without raw_image)
                import numpy as np
                img_tensor = example["pixel_values"]
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = (
                    img_np * np.array([0.229, 0.224, 0.225])
                    + np.array([0.485, 0.456, 0.406])
                )
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

            # Normalise bboxes using the ACTUAL image dimensions (not a
            # hardcoded constant) so that loc tokens span the full 0-999
            # range regardless of the spatial-transform resolution.
            img_w, img_h = pil_img.size

            # ── Build the target output text the model should generate ──
            detection_strs = []
            for bbox, label in zip(bboxes, labels):
                loc_tokens = _bbox_coco_to_florence(
                    bbox.tolist(), img_w, img_h
                )
                detection_strs.append(f"{label}{loc_tokens}")

            target_text = "".join(detection_strs) if detection_strs else ""

            images.append(pil_img)
            prompts.append("<OD>")
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
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
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
            dataloader_prefetch_factor=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            torch_compile=True,
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
    def inference(self, image: Image.Image, query: str = "") -> Dict:
        """Run object detection on a single image.

        Uses the ``<OD>`` task token (standard object detection) to match
        the training prompt.  The *query* parameter is accepted for API
        compatibility but ignored by the fine-tuned model.
        """
        prompt = "<OD>"

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
            task="<OD>",
            image_size=(image.width, image.height),
        )
        return parsed
