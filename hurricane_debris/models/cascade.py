"""
Cascaded Inference Pipeline
===========================
Florence-2 (detection) → SAM2 (segmentation) → structured JSON output.

This the core deliverable of the project: an end-to-end cascaded pipeline
for debris detection and segmentation from UAV imagery.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from hurricane_debris.config import DEBRIS_CATEGORIES, ExperimentConfig
from hurricane_debris.utils.logging import get_logger

logger = get_logger("models.cascade")


@dataclass
class Detection:
    """A single detected debris instance."""
    bbox: List[float]       # [x1, y1, x2, y2] absolute pixel coords
    category: str           # e.g. "building_damaged"
    score: float            # confidence score from Florence-2
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # H×W binary mask
    priority: str = "medium"  # "critical", "high", "medium", "low"


@dataclass
class InferenceResult:
    """Full result for one image."""
    image_path: str
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)

    def to_json(self) -> Dict:
        """Export as JSON-serialisable dict (masks excluded for size)."""
        return {
            "image": self.image_path,
            "width": self.width,
            "height": self.height,
            "num_detections": len(self.detections),
            "detections": [
                {
                    "bbox": d.bbox,
                    "category": d.category,
                    "score": round(d.score, 4),
                    "priority": d.priority,
                    "mask_available": d.mask is not None,
                }
                for d in self.detections
            ],
        }


# Priority rules based on debris taxonomy
_PRIORITY_MAP = {
    "building_damaged": "critical",
    "road_damaged": "critical",
    "water": "high",
    "vehicle": "high",
    "vegetation": "medium",
    "building_no_damage": "low",
    "road_no_damage": "low",
}


class CascadedInference:
    """
    End-to-end cascaded pipeline:
      1. Florence-2 open-vocabulary detection  →  bounding boxes
      2. Coordinate transform for SAM2 compatibility
      3. SAM2 mask generation per detected box
      4. Priority-based filtering via debris taxonomy
      5. Structured JSON report output
    """

    def __init__(
        self,
        florence_model_dir: str,
        sam2_checkpoint: str,
        config: Optional[ExperimentConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or ExperimentConfig()
        self.device = device or self.config.resolve_device()

        self.florence_model = None
        self.sam2_model = None
        self.florence_processor = None

        self._load_florence(florence_model_dir)
        self._load_sam2(sam2_checkpoint)

    # ── Model loading ────────────────────────────────────────────────────

    def _load_florence(self, model_dir: str):
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("Loading fine-tuned Florence-2 from %s", model_dir)
        self.florence_processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, trust_remote_code=True
        ).to(self.device).eval()

    def _load_sam2(self, checkpoint: str):
        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install from: "
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        logger.info("Loading fine-tuned SAM2 from %s", checkpoint)
        self.sam2_model = build_sam2(
            self.config.sam2.model_cfg, checkpoint
        ).to(self.device).eval()

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        query: str = "debris, damaged building, flooded area, downed tree, damaged road, vehicle wreckage",
    ) -> List[Detection]:
        """
        Stage 1: Run Florence-2 open-vocabulary detection.

        Returns list of Detection objects with bboxes and category names.
        """
        prompt = f"<OPEN_VOCABULARY_DETECTION>{query}"

        inputs = self.florence_processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.config.florence2.max_new_tokens,
            num_beams=self.config.florence2.num_beams,
        )

        text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.florence_processor.post_process_generation(
            text,
            task="<OPEN_VOCABULARY_DETECTION>",
            image_size=(image.width, image.height),
        )

        detections = []
        det_data = parsed.get("<OPEN_VOCABULARY_DETECTION>", {})
        bboxes = det_data.get("bboxes", [])
        labels = det_data.get("bboxes_labels", det_data.get("labels", []))
        scores = det_data.get("scores", [1.0] * len(bboxes))

        for bbox, label, score in zip(bboxes, labels, scores):
            cat = self._normalize_category(label)
            detections.append(
                Detection(
                    bbox=bbox,
                    category=cat,
                    score=float(score),
                    priority=_PRIORITY_MAP.get(cat, "medium"),
                )
            )

        logger.info("Florence-2 detected %d objects", len(detections))
        return detections

    @torch.no_grad()
    def segment(
        self, image: Image.Image, detections: List[Detection]
    ) -> List[Detection]:
        """
        Stage 2: Run SAM2 segmentation for each detected bounding box.

        Updates each Detection with a pixel-accurate mask.
        """
        if not detections:
            return detections

        import torch.nn.functional as F

        img_np = np.array(image)
        img_tensor = (
            torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        ).to(self.device)

        # Image embedding (shared across all boxes)
        img_input = img_tensor.unsqueeze(0)
        image_embedding = self.sam2_model.image_encoder(img_input)

        if isinstance(image_embedding, dict):
            backbone_out = image_embedding
        else:
            backbone_out = {"vision_features": image_embedding}

        if hasattr(self.sam2_model, '_prepare_backbone_features'):
            _, vision_feats, vision_pos_embeds, feat_sizes = (
                self.sam2_model._prepare_backbone_features(backbone_out)
            )
            B = 1
            image_embed = vision_feats[-1].reshape(B, -1, feat_sizes[-1][0], feat_sizes[-1][1])
            high_res_feats = [
                vf.reshape(B, -1, fs[0], fs[1])
                for vf, fs in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            image_embed = backbone_out if torch.is_tensor(backbone_out) else backbone_out["vision_features"]
            high_res_feats = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            box_coords = torch.tensor(
                [[x1, y1, x2, y2]], dtype=torch.float32, device=self.device
            )

            sparse_emb, dense_emb = self.sam2_model.sam_prompt_encoder(
                points=None, boxes=box_coords, masks=None
            )

            low_res_masks, iou_pred = self.sam2_model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
                high_res_features=high_res_feats if high_res_feats else None,
            )

            best_idx = iou_pred.argmax(dim=-1)
            best_mask = low_res_masks[0, best_idx]

            mask_full = F.interpolate(
                best_mask.unsqueeze(0),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()

            det.mask = (mask_full > 0).astype(np.uint8)

        logger.info("SAM2 generated masks for %d detections", len(detections))
        return detections

    def run(
        self,
        image_path: str,
        query: Optional[str] = None,
        score_threshold: float = 0.3,
    ) -> InferenceResult:
        """
        Full cascaded pipeline on a single image.

        Args:
            image_path: Path to UAV image.
            query: Open-vocabulary query string. Defaults to general debris query.
            score_threshold: Minimum confidence to keep a detection.

        Returns:
            InferenceResult with detections and masks.
        """
        image = Image.open(image_path).convert("RGB")

        # Stage 1: Detect
        detections = self.detect(image, query=query or (
            "debris, damaged building, flooded area, downed tree, "
            "damaged road, vehicle wreckage"
        ))

        # Filter by score
        detections = [d for d in detections if d.score >= score_threshold]

        # Stage 2: Segment
        detections = self.segment(image, detections)

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        detections.sort(key=lambda d: priority_order.get(d.priority, 9))

        result = InferenceResult(
            image_path=str(image_path),
            width=image.width,
            height=image.height,
            detections=detections,
        )

        logger.info(
            "Pipeline complete: %d detections on %s",
            len(detections), Path(image_path).name,
        )
        return result

    def run_batch(
        self,
        image_paths: List[str],
        output_json: Optional[str] = None,
        **kwargs,
    ) -> List[InferenceResult]:
        """Run the cascade on a batch of images and optionally save JSON."""
        results = []
        for path in image_paths:
            try:
                results.append(self.run(path, **kwargs))
            except Exception as e:
                logger.error("Failed on %s: %s", path, e)

        if output_json:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(
                    [r.to_json() for r in results], f, indent=2
                )
            logger.info("Results saved to %s", output_json)

        return results

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_category(raw_label: str) -> str:
        """Map Florence-2 free-form label to a canonical category name."""
        label = raw_label.lower().strip()
        mapping = {
            "debris": "building_damaged",
            "damaged building": "building_damaged",
            "collapsed building": "building_damaged",
            "building": "building_no_damage",
            "flooded area": "water",
            "flood": "water",
            "water": "water",
            "downed tree": "vegetation",
            "tree": "vegetation",
            "vegetation": "vegetation",
            "damaged road": "road_damaged",
            "road": "road_no_damage",
            "vehicle wreckage": "vehicle",
            "vehicle": "vehicle",
            "car": "vehicle",
        }
        for key, val in mapping.items():
            if key in label:
                return val
        return "building_damaged"  # conservative default
