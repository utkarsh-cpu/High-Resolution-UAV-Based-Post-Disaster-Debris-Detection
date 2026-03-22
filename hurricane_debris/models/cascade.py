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
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional for lightweight environments
    class _TorchStub:
        @staticmethod
        def no_grad():
            def _decorator(fn):
                return fn

            return _decorator

    torch = _TorchStub()
from PIL import Image

from hurricane_debris.config import DEBRIS_CATEGORIES, ExperimentConfig
from hurricane_debris.models.florence2 import load_florence_processor, _fix_florence2_weight_tying
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

    def to_geojson(self) -> Dict:
        """Export detections as pixel-coordinate GeoJSON polygons."""
        features = []
        for i, det in enumerate(self.detections):
            x1, y1, x2, y2 = [float(v) for v in det.bbox]
            polygon = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
                [x1, y1],
            ]
            features.append(
                {
                    "type": "Feature",
                    "id": i,
                    "geometry": {"type": "Polygon", "coordinates": [polygon]},
                    "properties": {
                        "category": det.category,
                        "score": round(det.score, 4),
                        "priority": det.priority,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "coordinate_system": "image_pixel",
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "coordinate_system": "image_pixel",
            "image": self.image_path,
            "image_size": {"width": self.width, "height": self.height},
            "features": features,
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


def _patch_florence2_config():
    """Patch cached Florence-2 remote code for Transformers 5.x compatibility.

    Fixes three issues in the HuggingFace-cached Florence-2 code:

    1. ``Florence2LanguageConfig`` accesses ``self.forced_bos_token_id``
       after ``super().__init__()``, but Transformers 5.x no longer
       creates attributes for ``None``-valued kwargs → use ``getattr``.

    2. ``Florence2ForConditionalGeneration`` defines ``_supports_sdpa``
       as a ``@property`` that reads ``self.language_model``, but
       Transformers 5.x checks it during ``__init__`` before the
       sub-model exists → add a class-level fallback attribute.

    3. ``past_key_values[0][0].shape[2]`` fails in Transformers 5.x
       because ``past_key_values`` is now an ``EncoderDecoderCache``
       object instead of a tuple → use ``get_seq_length()`` when available.
    """
    import glob, os

    base = "/home/.cache/huggingface/modules/transformers_modules"
    cfg_pattern = os.path.join(base, "**", "configuration_florence2.py")
    mdl_pattern = os.path.join(base, "**", "modeling_florence2.py")

    for path in glob.glob(cfg_pattern, recursive=True):
        with open(path) as f:
            src = f.read()
        old_check = "if self.forced_bos_token_id is None and kwargs.get("
        new_check = "if getattr(self, 'forced_bos_token_id', None) is None and kwargs.get("
        if old_check in src:
            src = src.replace(old_check, new_check)
            with open(path, "w") as f:
                f.write(src)
            logger.info("Patched %s (forced_bos_token_id)", path)

    for path in glob.glob(mdl_pattern, recursive=True):
        with open(path) as f:
            src = f.read()
        changed = False

        # Add class-level _supports_sdpa = True so it's available during __init__
        marker = "class Florence2ForConditionalGeneration(Florence2PreTrainedModel):\n"
        patched_marker = (
            "class Florence2ForConditionalGeneration(Florence2PreTrainedModel):\n"
            "    _supports_sdpa = True\n"
        )
        if marker in src and "_supports_sdpa = True\n    _tied_weights_keys" not in src:
            src = src.replace(marker, patched_marker)
            changed = True

        # Fix torch.linspace().item() on meta tensors
        old_linspace = "[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]"
        new_linspace = "torch.linspace(0, drop_path_rate, sum(depths)*2, device='cpu').tolist()"
        if old_linspace in src:
            src = src.replace(old_linspace, new_linspace)
            changed = True

        # Fix past_key_values subscript for EncoderDecoderCache in Transformers 5.x
        old_past = "past_key_values[0][0].shape[2]"
        new_past = "(past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else past_key_values[0][0].shape[2])"
        if old_past in src:
            src = src.replace(old_past, new_past)
            changed = True

        # Fix past_key_values_length subscript in decoder forward
        old_past_len = "past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0"
        new_past_len = "past_key_values_length = (past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else past_key_values[0][0].shape[2]) if past_key_values is not None else 0"
        if old_past_len in src:
            src = src.replace(old_past_len, new_past_len)
            changed = True

        if changed:
            with open(path, "w") as f:
                f.write(src)
            logger.info("Patched %s for Transformers 5.x compat", path)

    # Invalidate cached imports so Python re-reads the patched files
    import sys
    to_remove = [k for k in sys.modules if "florence2" in k.lower() and "transformers_modules" in k]
    for k in to_remove:
        del sys.modules[k]

    # Remove stale .pyc files
    for pyc in glob.glob(os.path.join(base, "**", "*.pyc"), recursive=True):
        if "florence2" in pyc.lower():
            os.remove(pyc)


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
        from pathlib import Path

        logger.info("Loading fine-tuned Florence-2 from %s", model_dir)
        self.florence_processor = load_florence_processor(model_dir)

        _patch_florence2_config()

        adapter_config = Path(model_dir) / "adapter_config.json"
        if adapter_config.exists():
            # LoRA adapter directory — load base model then merge adapter
            import json
            with open(adapter_config) as f:
                acfg = json.load(f)
            base_model_id = acfg.get("base_model_name_or_path", "microsoft/Florence-2-base-ft")
            base_model = self._load_florence_base(base_model_id)
            from peft import PeftModel
            self.florence_model = PeftModel.from_pretrained(
                base_model, model_dir
            ).merge_and_unload().to(self.device).eval()
        else:
            # Full model checkpoint
            self.florence_model = self._load_florence_base(model_dir).to(self.device).eval()

    @staticmethod
    def _load_florence_base(model_id: str):
        """Load a Florence-2 base model, patching config for Transformers 5.x compat."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        )
        _fix_florence2_weight_tying(model)
        return model

    def _load_sam2(self, checkpoint: str):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install from: "
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        logger.info("Loading fine-tuned SAM2 from %s", checkpoint)
        state_dict = torch.load(checkpoint, map_location=self.device)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        # Strip torch.compile _orig_mod. prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("._orig_mod.", ".")] = v

        self.sam2_model = build_sam2(
            self.config.sam2.model_cfg
        ).to(self.device)
        self.sam2_model.load_state_dict(cleaned)
        self.sam2_model.eval()
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

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
        # The fine-tuned model uses the <OD> (object detection) task token,
        # which activates proper vision-language grounding for precise bbox
        # prediction.  Using <OPEN_VOCABULARY_DETECTION> without a query
        # disables spatial grounding and produces degenerate coordinates.
        prompt = "<OD>"

        inputs = self.florence_processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.config.florence2.max_new_tokens,
            num_beams=self.config.florence2.num_beams,
            use_cache=False,
        )

        text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        logger.info("Florence-2 raw output: %s", text[:500])

        parsed = self.florence_processor.post_process_generation(
            text,
            task="<OD>",
            image_size=(image.width, image.height),
        )

        logger.info("Florence-2 parsed result: %s", parsed)

        detections = []
        det_data = parsed.get("<OD>", {})
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

        # Use SAM2ImagePredictor which handles image transforms, feature
        # extraction (including no_mem_embed), and proper box→point prompt
        # conversion internally.
        self.sam2_predictor.set_image(image)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            masks, iou_pred, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=True,
            )

            best_idx = int(iou_pred.argmax())
            det.mask = masks[best_idx].astype(np.uint8)

        self.sam2_predictor.reset_predictor()
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

        # Drop degenerate bboxes (zero or near-zero area)
        min_side = 2.0  # minimum bbox side in pixels
        detections = [
            d for d in detections
            if abs(d.bbox[2] - d.bbox[0]) >= min_side
            and abs(d.bbox[3] - d.bbox[1]) >= min_side
        ]

        # Stage 2: Segment
        detections = self.segment(image, detections)

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        detections.sort(
            key=lambda d: (
                priority_order.get(d.priority, 9),
                -float(d.score),
                d.category,
                tuple(float(v) for v in d.bbox),
            )
        )

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

        # Exact matches for training-time CATEGORY_QUERIES labels first
        exact = {
            "flooded area with standing water": "water",
            "intact undamaged building": "building_no_damage",
            "damaged or collapsed building with debris": "building_damaged",
            "vegetation and downed trees": "vegetation",
            "intact undamaged road": "road_no_damage",
            "damaged road with cracks or debris": "road_damaged",
            "vehicle or vehicle wreckage": "vehicle",
        }
        if label in exact:
            return exact[label]

        # Substring matches — more specific phrases checked before generic ones
        mapping = [
            ("damaged road", "road_damaged"),
            ("damaged building", "building_damaged"),
            ("collapsed building", "building_damaged"),
            ("vehicle wreckage", "vehicle"),
            ("downed tree", "vegetation"),
            ("flooded area", "water"),
            ("debris", "building_damaged"),
            ("flood", "water"),
            ("water", "water"),
            ("vegetation", "vegetation"),
            ("tree", "vegetation"),
            ("road", "road_no_damage"),
            ("building", "building_no_damage"),
            ("vehicle", "vehicle"),
            ("car", "vehicle"),
        ]
        for key, val in mapping:
            if key in label:
                return val
        return "building_damaged"  # conservative default
