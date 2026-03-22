# High-Resolution UAV-Based Post-Disaster Debris Detection

A modular cascaded AI pipeline for **debris detection and pixel-level
segmentation** from high-resolution UAV imagery. The system fine-tunes two
foundation models and chains them at inference time:

1. **Florence-2** (Microsoft) — open-vocabulary object detection via LoRA
   fine-tuning
2. **SAM2 Hiera-Large** (Meta AI) — instance segmentation with frozen image
   encoder and trainable prompt/mask decoders

The pipeline supports **three multi-source UAV datasets** (RescueNet, MSNet,
DesignSafe-CI), a unified **8-class debris taxonomy**, cross-dataset
evaluation, and structured export in JSON and image-coordinate GeoJSON
formats. A **Gradio web interface** is included for interactive demo
inference.

## Repository Layout

```text
.
├── app.py                              # Gradio web demo for local inference
├── main.py                             # Main CLI: download / train / evaluate / infer
├── demo.py                             # Standalone demo script (2-3 sample images)
├── hurricane_debris/                   # Core Python package
│   ├── config.py                       #   Central configuration dataclasses
│   ├── models/
│   │   ├── cascade.py                  #   Cascaded inference (Florence-2 → SAM2)
│   │   ├── florence2.py                #   Florence-2 LoRA trainer + Transformers 5.x patches
│   │   └── sam2_trainer.py             #   SAM2 partial fine-tuning trainer
│   ├── data/
│   │   ├── base_dataset.py             #   COCO-format base dataset class
│   │   ├── rescuenet.py                #   RescueNet loader (colour-mask auto-decoding)
│   │   ├── msnet.py                    #   MSNet/ISBDA loader (oriented bbox support)
│   │   ├── designsafe.py              #   DesignSafe-CI PRJ-6029 loader
│   │   ├── transforms.py              #   Albumentations augmentation pipeline
│   │   ├── splits.py                   #   Stratified train/val/test splitting
│   │   └── download.py                #   Dataset download & archive extraction
│   ├── evaluation/
│   │   └── metrics.py                  #   Evaluator (mIoU, F1, AP@[0.5:0.95])
│   ├── utils/
│   │   └── logging.py                  #   Structured logging setup
│   └── tests/                          #   pytest test suite
├── scripts/
│   ├── run_experiments.py              # Experiment matrix runner & aggregator
│   └── experiment_matrix.json          # Experiment definitions
├── checkpoints/                        # Pre-trained SAM2 checkpoint
├── models/                             # Fine-tuned model outputs
│   ├── florence2_debris/               #   Florence-2 LoRA adapter + processor
│   └── sam2_debris/                    #   SAM2 best & final checkpoints
├── outputs/metrics/                    # Evaluation metric JSON files
├── requirements.txt                    # Python dependencies
└── PROJECT_V3.md                       # Implementation status & scope notes
```

## Features

- **Cascaded detection → segmentation** pipeline (Florence-2 → SAM2)
- **LoRA fine-tuning** for Florence-2 (0.8% parameter overhead, rank 16)
- **Partial SAM2 fine-tuning** (frozen image encoder, trainable prompt + mask
  decoder)
- **Batched SAM2 inference** — single image encoder call + batched mask decoder
  per image
- **AMP (mixed precision)** with bfloat16/float16 autocast and `torch.compile`
  on mask decoder
- **Albumentations augmentation** pipeline with bbox-aware spatial transforms
- **Three dataset loaders** with automatic layout detection and format
  conversion
- **8-class unified debris taxonomy** with priority-based output sorting
- **Evaluation metrics**: mIoU (per-class), F1, Precision, Recall,
  AP@50/75/[0.5:0.95]
- **Cross-dataset benchmarking** (train on RescueNet, test on MSNet/DesignSafe)
- **Experiment matrix** runner for baselines and ablations (JSON + CSV output)
- **Deterministic reproducibility** via `--seed` with full torch/numpy/random
  synchronisation
- **Gradio web interface** for interactive inference
- **Structured export** in JSON and image-coordinate GeoJSON

## Setup

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.10 | Runtime |
| PyTorch | >= 2.1.0 | Deep learning backend |
| Transformers | >= 4.38.0 | Florence-2 model |
| PEFT | >= 0.8.0 | LoRA adapters |
| Albumentations | >= 1.3.1 | Data augmentation |
| Gradio | >= 4.44.0 | Web demo |
| pycocotools | >= 2.0.7 | COCO evaluation |

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

Install Git LFS before cloning or pulling model/checkpoint updates:

```bash
git lfs install
git lfs pull
```

Install **SAM2** separately (not on PyPI):

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Device Selection

The CLI flag `--device` controls hardware acceleration:

| Value | Behaviour |
|-------|-----------|
| `auto` (default) | CUDA if available → MPS if available → CPU |
| `cuda` | Force NVIDIA GPU |
| `mps` | Force Apple Silicon GPU |
| `cpu` | Force CPU |

GPU acceleration is strongly recommended for training and large-scale
inference.

## Debris Taxonomy (8 Classes)

All datasets are mapped to a unified taxonomy defined in
`hurricane_debris/config.py`:

| ID | Class | Priority | Description |
|----|-------|----------|-------------|
| 0 | `background` | — | Non-debris / unlabelled |
| 1 | `water` | high | Flooded areas, standing water |
| 2 | `building_no_damage` | low | Intact structures |
| 3 | `building_damaged` | **critical** | Collapsed / damaged buildings with debris |
| 4 | `vegetation` | medium | Downed trees, vegetation debris |
| 5 | `road_no_damage` | low | Clear roads |
| 6 | `road_damaged` | **critical** | Blocked / cracked roads |
| 7 | `vehicle` | high | Vehicles or wreckage |

Detections are sorted by priority → confidence → category for actionable
output.

## Datasets

The project supports three complementary UAV datasets.

### RescueNet (Primary)

**Source:** Alam et al., IEEE TGRS 2022 — Hurricane Michael (FL, 2018)
**Size:** 4,494 images at 0.5–2 cm/px GSD
**Annotation:** Pixel-level semantic masks (8 classes)
**Splits:** Official train / val / test provided

The loader auto-detects two directory layouts:

- **Flat layout:** images and masks under a single `RescueNet/` root
- **Dropbox layout:** `RescueNet/` and `ColorMasks-RescueNet/` as siblings

Colour-mask variants (`ColorMasks-RescueNet`, `ColourMasks-RescueNet`,
`colourmask-rescuenet`) are auto-decoded via an RGB palette lookup. The
official 11-class release is remapped to the 8-class taxonomy (e.g.
Building-Minor/Major/Total-Destruction → `building_damaged`). Instance
bounding boxes are extracted via connected-component analysis
(`min_component_area=100`).

### MSNet / ISBDA (Cross-dataset)

**Source:** Xie & Xiong, Remote Sensing 2023; Zhu et al., WACV 2021
**Size:** 8,700+ images at 0.3–5 cm/px across 7 disaster events
**Annotation:** COCO-format instances with damage levels and oriented bboxes
**Splits:** train / val (test falls back to val if unavailable)

Oriented bounding boxes (DOTA format) are automatically converted to
axis-aligned boxes. Damage level mapping: Slight (1), Severe (2), Debris (3)
→ all map to `building_damaged`. The loader uses a fallback hierarchy for
bounding boxes: `bbox` → `damage_bbox` → `house_bbox` → `oriented_bbox`.

### DesignSafe-CI PRJ-6029 (Cross-dataset)

**Source:** Amini et al., NSF NHERI DesignSafe-CI — Hurricanes Ian, Ida, Ike
**Size:** 1,242 images (508 with masks)
**Annotation:** 3-class binary masks (no debris / low density / high density)
**Splits:** Stratified 70/15/15 (images with/without masks split separately)

Both V2 (preferred) and V1 directory layouts are auto-detected. Low/high
density debris both map to `building_damaged`.
Images without masks are treated as debris-free background-only samples.
Empty samples are filtered before training via `has_foreground` scan
(`min_component_area=50`).

### Download

```bash
# Download individual datasets
python main.py --download --dataset rescuenet --dataset-dir ./datasets
python main.py --download --dataset msnet --dataset-dir ./datasets
python main.py --download --dataset designsafe --dataset-dir ./datasets

# Download all at once
python main.py --download --dataset all --dataset-dir ./datasets

# Force re-download / keep archive after extraction
python main.py --download --dataset rescuenet --dataset-dir ./datasets --force-download --keep-archive
```

If a matching archive already exists under `--dataset-dir`, the download
helper reuses it. Supported local archive names:

- RescueNet: `rescuenet.zip`, or `RescueNet.zip` + `ColorMasks-RescueNet.zip`
  (bundle)
- MSNet: `msnet.zip` or `ISBDA.zip`
- DesignSafe: `designsafe.zip` or `PRJ-6029.zip`

Download uses `gdown` for Google Drive mirrors and direct HTTPS for other
URLs. When automatic download is unavailable, the CLI prints manual
acquisition instructions.

### Data Augmentation

The augmentation pipeline (`hurricane_debris/data/transforms.py`) uses
Albumentations with bbox-aware spatial transforms:

**Training:**
- `RandomResizedCrop` (scale 0.8–1.0)
- `HorizontalFlip` (p=0.5), `VerticalFlip` (p=0.2), `RandomRotate90`
  (p=0.5)
- `ColorJitter` (brightness/contrast/saturation=0.2, p=0.3)
- `GaussNoise` (std 10–50/255, p=0.2)
- ImageNet normalisation: mean=(0.485, 0.456, 0.406),
  std=(0.229, 0.224, 0.225)

**Validation/Test:** Resize only, no random transforms.

Bounding box handling: COCO format with `min_visibility=0.3` (nearly-clipped
boxes removed). Raw PIL images are preserved alongside tensors to avoid lossy
denormalise→renormalise round-trips during Florence-2 collation.

## Quick Start

### Full pipeline (train Florence-2 + SAM2, then evaluate)

```bash
python main.py --full-pipeline --dataset rescuenet --dataset-dir ./datasets
```

### Train individual stages

```bash
# Florence-2 detection training with custom hyperparameters
python main.py --train-florence --dataset rescuenet --dataset-dir ./datasets \
    --epochs-florence 10 --lr-florence 5e-5 --batch-size 4 --image-size 768

# SAM2 segmentation training
python main.py --train-sam2 --dataset rescuenet --dataset-dir ./datasets \
    --epochs-sam2 20 --lr-sam2 1e-5

# Train on all datasets combined (ConcatDataset)
python main.py --train-florence --dataset all --dataset-dir ./datasets
```

### Evaluate

```bash
# Single dataset evaluation
python main.py --evaluate --dataset rescuenet --dataset-dir ./datasets \
    --metrics-dir ./outputs/metrics

# Cross-dataset evaluation (train RescueNet → test DesignSafe + MSNet)
python main.py --evaluate --dataset rescuenet --dataset-dir ./datasets \
    --cross-dataset --metrics-dir ./outputs/metrics

# Strict mode: fail if cascade models cannot load (vs graceful fallback)
python main.py --evaluate --dataset rescuenet --strict-eval-model
```

### Run inference

```bash
python main.py --infer --image path/to/image.jpg
python main.py --infer --image path/to/image.jpg --output-json out.json
python main.py --infer --image path/to/image.jpg --output-json out.json --output-geojson out.geojson
```

## CLI Reference

All arguments accepted by `main.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--download` | — | Download and extract datasets |
| `--full-pipeline` | — | Run complete train + evaluate pipeline |
| `--train-florence` | — | Train Florence-2 only |
| `--train-sam2` | — | Train SAM2 only |
| `--evaluate` | — | Evaluate trained models |
| `--infer` | — | Run inference on a single image |
| `--dataset` | `rescuenet` | Dataset: `rescuenet`, `msnet`, `designsafe`, or `all` |
| `--dataset-dir` | `./datasets` | Root directory for datasets |
| `--florence-dir` | `./models/florence2_debris` | Florence-2 model output directory |
| `--sam2-dir` | `./models/sam2_debris` | SAM2 model output directory |
| `--sam2-checkpoint` | `./checkpoints/sam2_hiera_large.pt` | SAM2 pre-trained checkpoint |
| `--epochs-florence` | `10` | Florence-2 training epochs |
| `--epochs-sam2` | `20` | SAM2 training epochs |
| `--lr-florence` | `5e-5` | Florence-2 learning rate |
| `--lr-sam2` | `1e-5` | SAM2 learning rate |
| `--batch-size` | `4` | Training batch size |
| `--image-size` | `768` | Input image resolution |
| `--device` | `auto` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `--seed` | `42` | Random seed for reproducibility |
| `--metrics-dir` | `./outputs/metrics` | Evaluation output directory |
| `--cross-dataset` | — | Enable cross-dataset evaluation |
| `--strict-eval-model` | — | Fail if cascade models cannot load |
| `--image` | — | Image path for `--infer` |
| `--output-json` | — | JSON output path for `--infer` |
| `--output-geojson` | — | GeoJSON output path for `--infer` |
| `--force-download` | — | Re-download even if data exists |
| `--keep-archive` | — | Keep archive after extraction |
| `--log-file` | `./logs/training.log` | Log file path |

## Training Details

### Florence-2 (Detection)

- **Base model:** `microsoft/Florence-2-base-ft`
- **Adaptation:** LoRA with r=16, alpha=16, dropout=0.1
- **LoRA target modules:** `q_proj`, `v_proj`, `k_proj`, `o_proj` (attention
  heads)
- **Task token:** `<OD>` (object detection mode)
- **Optimiser:** AdamW (lr=5e-5, weight_decay=0.01)
- **Scheduler:** Linear warmup (10% of steps) + cosine decay
- **Batch:** 8 effective (batch_size=4 × gradient_accumulation=2)
- **Precision:** bfloat16 on Ampere+ GPUs, float16 fallback
- **Early stopping:** Patience 3 epochs on validation loss
- **Generation:** max_new_tokens=512, num_beams=3

Collation preserves raw PIL images (avoids lossy denormalise→renormalise
round-trips). The trainer includes **Transformers 5.x compatibility patches**
for `forced_bos_token_id`, `_supports_sdpa`, `past_key_values`
EncoderDecoderCache, and `torch.linspace` on meta tensors.

### SAM2 (Segmentation)

- **Base model:** SAM2 Hiera-Large (1.3B parameters)
- **Strategy:** Image encoder frozen; prompt encoder + mask decoder trainable
- **Loss:** Dice + BCE (weight 1.0 each)
- **Optimiser:** AdamW (lr=1e-5)
- **Scheduler:** Cosine annealing with 10% linear warmup
- **Batch:** 8 effective (batch_size=4 × gradient_accumulation=2)
- **Early stopping:** Patience 5 epochs on validation loss
- **Multimask output:** 3 masks generated, best selected by predicted IoU

**Performance optimisations:**
- Batched image encoding (one forward pass per batch, not per image)
- Batched box prompts per image (one decoder call per image, not per box)
- AMP (mixed precision) with bfloat16 autocast
- `torch.compile()` on mask decoder for fused CUDA kernels
- Gradient checkpointing on mask decoder for VRAM efficiency
- cuDNN benchmark mode for fixed-size inputs

### Reproducibility

Deterministic training is controlled via `--seed` (default 42):
- `torch.manual_seed()` + `torch.cuda.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `random.seed()` + `numpy.random.seed()` synchronised
- Full `ExperimentConfig` saved to `outputs/metrics/run_config.json`

## Cascaded Inference Pipeline

The core deliverable is a two-stage cascade in
`hurricane_debris/models/cascade.py`:

```
Input Image
  │
  ▼
Florence-2 (LoRA) ──► Bounding boxes + category labels + confidence scores
  │
  ▼
Score threshold filter + bbox validity check
  │
  ▼
SAM2 (fine-tuned) ──► Per-object binary masks (H×W)
  │
  ▼
Priority sort ──► Structured JSON + GeoJSON output
```

Each detection includes: `bbox` (xyxy pixels), `category`, `score`,
`priority`, and optional `mask`. Outputs exclude masks in JSON for size;
GeoJSON exports bounding-box polygons in image-pixel coordinates.

> **GeoJSON scope:** export uses image-pixel coordinates. Full
> GIS-georeferenced reprojection requires camera pose / orthorectification
> metadata not available in current datasets.

## Model and Output Paths

| Path | Contents |
|------|----------|
| `./models/florence2_debris/` | LoRA adapter, processor, tokeniser |
| `./models/sam2_debris/` | `best_model.pth`, `final_model.pth` |
| `./checkpoints/sam2_hiera_large.pt` | Pre-trained SAM2 Hiera-Large |
| `./outputs/metrics/` | Per-dataset metric JSONs, `run_config.json` |

## Evaluation Metrics

The evaluator (`hurricane_debris/evaluation/metrics.py`) computes:

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection-over-Union (per-class, excluding background) |
| **Per-class IoU** | Individual IoU for each of the 8 classes |
| **F1** | Detection F1 at IoU threshold (default 0.5) |
| **Precision / Recall** | Detection-level metrics |
| **AP@50** | Average Precision at IoU ≥ 0.5 |
| **AP@75** | Average Precision at IoU ≥ 0.75 |
| **AP@[0.5:0.95]** | COCO-style AP across 10 IoU thresholds (0.5, 0.55, …, 0.95) |

Detection matching uses **IoU-based greedy assignment**. AP uses 11-point
interpolation. Confusion matrices are accumulated for semantic segmentation
mIoU.

Metrics are saved per dataset as `metrics_<dataset>.json` and optionally
as `cross_dataset_summary.json`.

## Baseline and Ablation Experiments

Run the experiment matrix and aggregate results:

```bash
python scripts/run_experiments.py \
    --matrix scripts/experiment_matrix.json \
    --output-dir ./outputs/experiments
```

The experiment matrix (`scripts/experiment_matrix.json`) defines:
- **Defaults:** dataset, dataset_dir, seed, cross_dataset flag
- **Experiments:** named configurations with custom CLI args

Output artifacts:
- `outputs/experiments/experiment_summary.json` — full structured results
- `outputs/experiments/experiment_summary.csv` — tabular summary for analysis
- `outputs/experiments/runs/<experiment_name>/` — per-experiment outputs

## Gradio Demo

Launch the local web app:

```bash
python app.py \
    --florence-dir ./models/florence2_debris \
    --sam2-checkpoint ./models/sam2_debris/best_model.pth \
    --device auto \
    --server-name 127.0.0.1 \
    --server-port 7860
```

Then open `http://127.0.0.1:7860`. The interface provides:
- **Image upload** (PNG, JPG, TIFF)
- **Open-vocabulary query** text input (default: "debris, damaged building,
  flooded area, downed tree, damaged road, vehicle wreckage")
- **Score threshold** slider (0.0–1.0, default 0.3)
- **Structured JSON** and **GeoJSON** output panels

If models fail to load, the app displays a warning but does not crash.

## Standalone Demo

Run inference on sample images with visualised output:

```bash
python demo.py
```

This processes 2–3 sample UAV images from the test set, prints detection JSON,
and saves annotated images to `outputs/demo/`.

## Validation and Tests

The test suite lives in `hurricane_debris/tests/`:

```bash
python -m pytest                                              # all tests
python -m pytest --cov                                        # with coverage
python -m pytest hurricane_debris/tests/test_cascade.py       # cascade only
python -m pytest hurricane_debris/tests/test_main_integration.py  # CLI integration
```

Test modules:
- `test_cascade.py` — Detection, InferenceResult, priority sorting, JSON/GeoJSON schema
- `test_datasets.py` — Dataset loaders, layout detection, class mapping
- `test_florence2.py` — Florence-2 collation, LoRA setup
- `test_sam2.py` — SAM2 module trainability, freeze strategy
- `test_evaluator.py` — Metric computation, AP threshold logic
- `test_main_integration.py` — End-to-end CLI integration
- `test_download.py` — Download helpers, archive extraction

If these commands fail with `No module named ...`, install dependencies first
with `pip install -r requirements.txt`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python main.py --help` fails with missing module | Run `pip install -r requirements.txt` first |
| SAM2 import errors | Install SAM2 separately: `pip install git+https://github.com/facebookresearch/segment-anything-2.git` |
| Checkpoint path errors during inference | Confirm `--florence-dir` and `--sam2-dir` match your training output directories |
| Dataset download limitations | Some datasets require registration; follow the CLI's manual instructions |
| Transformers 5.x compatibility warnings | The codebase auto-patches known issues; update to latest Transformers if problems persist |
| CUDA out-of-memory during SAM2 training | Reduce `--batch-size` or `--image-size`; gradient checkpointing is already enabled |
| DesignSafe loader skipping many images | Expected — only 508/1242 images have masks; empty samples are filtered |

## Logging

Structured logging via `hurricane_debris/utils/logging.py`:
- Format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`
- Console output + optional file logging via `--log-file` (default:
  `./logs/training.log`)
- Module-level loggers: `get_logger("models.cascade")`, etc.

## Project Scope Notes

- CLI-first workflow in `main.py`; Gradio app for interactive demos
- GeoJSON uses image-pixel coordinates (not GIS-georeferenced — requires
  camera pose metadata not provided in current datasets)
- See [`PROJECT_V3.md`](./PROJECT_V3.md) for implementation status and
  research-oriented scope clarifications

## References

1. M. S. Alam et al., "RescueNet: A High-Resolution Post-Disaster UAV Dataset
   for Semantic Segmentation," *IEEE Trans. Geosci. Remote Sens.*, vol. 60,
   2022.
2. K. Amini et al., "Hurricane-Induced Debris Segmentation Dataset Using
   Aerial Imagery" [v2], DesignSafe-CI, 2025.
   DOI: 10.17603/ds2-jvps-2n95
3. X. Zhu, J. Liang, and A. Hauptmann, "MSNet: A Multilevel Instance
   Segmentation Network for Natural Disaster Damage Assessment in Aerial
   Videos," *WACV*, pp. 2023–2032, 2021.
4. W. Xiao et al., "Florence-2: Advancing a Unified Representation for a
   Variety of Vision Tasks," *CVPR*, 2024.
5. N. Ravi et al., "SAM 2: Segment Anything in Images and Videos,"
   *arXiv:2408.00714*, 2024.
6. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models,"
   *ICLR*, 2022.
