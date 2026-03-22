#!/usr/bin/env python3
"""
Standalone Demo — Hurricane Debris Detection
=============================================
Runs the Florence-2 + SAM2 cascaded inference on 2-3 sample UAV images
from the test set, prints detection results, and saves annotated images
with bounding boxes and masks overlaid.

Usage:
    python demo.py
    python demo.py --device cuda --score-threshold 0.25
    python demo.py --images img1.jpg img2.jpg img3.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────

# Default sample images from the RescueNet test set
DEFAULT_SAMPLES = [
    "datasets/rescuenet/RescueNet/test/test-org-img/10794.jpg",
    "datasets/rescuenet/RescueNet/test/test-org-img/11358.jpg",
    "datasets/rescuenet/RescueNet/test/test-org-img/14898.jpg",
]

# Colour palette for visualisation (per-category BGR)
CATEGORY_COLOURS = {
    "water":              (255, 180, 0),    # cyan-blue
    "building_no_damage": (0, 200, 0),      # green
    "building_damaged":   (0, 0, 255),      # red
    "vegetation":         (0, 180, 0),      # dark green
    "road_no_damage":     (180, 180, 180),  # grey
    "road_damaged":       (0, 100, 255),    # orange-red
    "vehicle":            (255, 0, 180),    # magenta
    "background":         (128, 128, 128),  # grey
}

PRIORITY_LABELS = {
    "critical": "[CRITICAL]",
    "high":     "[HIGH]    ",
    "medium":   "[MEDIUM]  ",
    "low":      "[LOW]     ",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: Hurricane Debris Detection (Florence-2 + SAM2)"
    )
    parser.add_argument(
        "--images", nargs="+", default=None,
        help="Paths to input images. Defaults to 3 RescueNet test images.",
    )
    parser.add_argument("--florence-dir", default="./models/florence2_debris")
    parser.add_argument("--sam2-checkpoint", default="./models/sam2_debris/best_model.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", default="./outputs/demo")
    return parser.parse_args()


def draw_detections(image_path: str, result, output_path: str):
    """Draw bounding boxes, masks, and labels on the image and save."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARNING] Could not read image for visualisation: {image_path}")
        return

    overlay = img.copy()

    for det in result.detections:
        colour = CATEGORY_COLOURS.get(det.category, (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in det.bbox]

        # Draw semi-transparent mask if available
        if det.mask is not None:
            mask_bool = det.mask.astype(bool)
            # Resize mask to image dimensions if needed
            if mask_bool.shape[:2] != img.shape[:2]:
                mask_resized = cv2.resize(
                    det.mask.astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask_resized = mask_bool
            overlay[mask_resized] = (
                np.array(colour) * 0.4 + overlay[mask_resized] * 0.6
            ).astype(np.uint8)

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        # Draw label background + text
        label = f"{det.category} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            overlay, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    cv2.imwrite(output_path, overlay)
    print(f"  Annotated image saved: {output_path}")


def print_result_summary(result):
    """Print a formatted summary of detections to console."""
    print(f"\n{'='*70}")
    print(f"  Image: {result.image_path}")
    print(f"  Size:  {result.width} x {result.height}")
    print(f"  Detections: {len(result.detections)}")
    print(f"{'='*70}")

    if not result.detections:
        print("  No detections above threshold.\n")
        return

    for i, det in enumerate(result.detections):
        priority_tag = PRIORITY_LABELS.get(det.priority, "[?]       ")
        bbox_str = f"[{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f}]"
        mask_str = "Yes" if det.mask is not None else "No"
        print(
            f"  {i+1:2d}. {priority_tag}  {det.category:<22s}  "
            f"score={det.score:.3f}  bbox={bbox_str}  mask={mask_str}"
        )
    print()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve sample images
    image_paths = args.images or DEFAULT_SAMPLES
    valid_paths = []
    for p in image_paths:
        if Path(p).exists():
            valid_paths.append(p)
        else:
            print(f"[WARNING] Image not found, skipping: {p}")

    if not valid_paths:
        print("[ERROR] No valid input images found.")
        print("  Provide images via: python demo.py --images img1.jpg img2.jpg")
        sys.exit(1)

    # Load the cascade pipeline
    print("\n" + "="*70)
    print("  Hurricane Debris Detection Demo")
    print("  Florence-2 (LoRA) + SAM2 (Hiera-Large) Cascade")
    print("="*70)
    print(f"\n  Loading models...")
    print(f"    Florence-2 : {args.florence_dir}")
    print(f"    SAM2       : {args.sam2_checkpoint}")
    print(f"    Device     : {args.device}")
    print(f"    Threshold  : {args.score_threshold}")
    print(f"    Images     : {len(valid_paths)}")
    print()

    from hurricane_debris.config import ExperimentConfig
    from hurricane_debris.models.cascade import CascadedInference

    config = ExperimentConfig(device=args.device)
    pipeline = CascadedInference(
        florence_model_dir=args.florence_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        config=config,
        device=config.resolve_device(),
    )

    print("  Models loaded successfully.\n")

    # Run inference on each image
    all_results = []
    for img_path in valid_paths:
        print(f"  Processing: {Path(img_path).name} ...")
        result = pipeline.run(
            img_path,
            score_threshold=args.score_threshold,
        )

        # Print human-readable summary
        print_result_summary(result)

        # Save annotated visualisation
        out_img = str(output_dir / f"{Path(img_path).stem}_annotated.jpg")
        draw_detections(img_path, result, out_img)

        # Save structured JSON
        out_json = str(output_dir / f"{Path(img_path).stem}_detections.json")
        with open(out_json, "w") as f:
            json.dump(result.to_json(), f, indent=2)
        print(f"  Detection JSON saved: {out_json}")

        # Save GeoJSON
        out_geojson = str(output_dir / f"{Path(img_path).stem}_detections.geojson")
        with open(out_geojson, "w") as f:
            json.dump(result.to_geojson(), f, indent=2)
        print(f"  GeoJSON saved: {out_geojson}")

        all_results.append(result)

    # Print overall summary
    total_dets = sum(len(r.detections) for r in all_results)
    print("\n" + "="*70)
    print(f"  Demo Complete")
    print(f"  Images processed : {len(all_results)}")
    print(f"  Total detections : {total_dets}")
    print(f"  Output directory : {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
