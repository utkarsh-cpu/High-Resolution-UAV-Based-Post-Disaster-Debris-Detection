"""Gradio app for Hurricane Debris cascaded inference."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from PIL import Image

from hurricane_debris.config import ExperimentConfig
from hurricane_debris.models.cascade import CascadedInference

# Per-category BGR colours for drawing
_CATEGORY_COLOURS = {
    "water":              (255, 180, 0),
    "building_no_damage": (0, 200, 0),
    "building_damaged":   (0, 0, 255),
    "vegetation":         (0, 180, 0),
    "road_no_damage":     (180, 180, 180),
    "road_damaged":       (0, 100, 255),
    "vehicle":            (255, 0, 180),
    "background":         (128, 128, 128),
    "debris":             (255, 255, 0),
}


def _draw_detections(pil_image, result):
    """Return a PIL image with bboxes, masks, and labels drawn."""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    overlay = img.copy()

    for det in result.detections:
        colour = _CATEGORY_COLOURS.get(det.category, (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in det.bbox]

        # Semi-transparent mask overlay
        if det.mask is not None:
            mask = det.mask.astype(np.uint8)
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (
                np.array(colour) * 0.4 + overlay[mask_bool] * 0.6
            ).astype(np.uint8)

        # Bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        # Label with background
        label = f"{det.category} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def _load_pipeline(florence_dir: str, sam2_checkpoint: str, device: str):
    config = ExperimentConfig(device=device)
    return CascadedInference(
        florence_model_dir=florence_dir,
        sam2_checkpoint=sam2_checkpoint,
        config=config,
        device=config.resolve_device(),
    )


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio app for cascaded debris inference")
    parser.add_argument("--florence-dir", default="./models/florence2_debris")
    parser.add_argument("--sam2-checkpoint", default="./models/sam2_debris/best_model.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    try:
        pipeline = _load_pipeline(args.florence_dir, args.sam2_checkpoint, args.device)
        init_error = ""
    except Exception as exc:
        pipeline = None
        init_error = f"Pipeline failed to load: {exc}"

    def infer(image, query, score_threshold):
        if image is None:
            return None, "Please upload an image.", ""
        if pipeline is None:
            return None, "", init_error

        temp_path = Path("./outputs/gradio")
        temp_path.mkdir(parents=True, exist_ok=True)
        image_path = temp_path / "input.png"
        image.save(image_path)

        result = pipeline.run(
            str(image_path),
            query=query.strip() if query else None,
            score_threshold=float(score_threshold),
        )

        annotated = _draw_detections(image, result)

        return (
            annotated,
            json.dumps(result.to_json(), indent=2),
            json.dumps(result.to_geojson(), indent=2),
        )

    with gr.Blocks(title="Hurricane Debris Detection Demo") as demo:
        gr.Markdown("# Hurricane Debris Detection Demo")
        gr.Markdown(
            "Florence-2 + SAM2 cascaded inference with annotated output, JSON, and image-pixel GeoJSON."
        )
        if init_error:
            gr.Markdown(f"**Warning:** {init_error}")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input UAV Image")
            with gr.Column():
                query_input = gr.Textbox(
                    label="Open-vocabulary Query",
                    value="debris, damaged building, flooded area, downed tree, damaged road, vehicle wreckage",
                )
                score_input = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Score Threshold")
                run_btn = gr.Button("Run Inference")

        annotated_out = gr.Image(type="pil", label="Annotated Output")
        json_out = gr.Code(label="Structured JSON", language="json")
        geojson_out = gr.Code(label="GeoJSON (image_pixel)", language="json")

        run_btn.click(
            fn=infer,
            inputs=[image_input, query_input, score_input],
            outputs=[annotated_out, json_out, geojson_out],
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
