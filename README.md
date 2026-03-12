# High-Resolution UAV-Based Post-Disaster Debris Detection

This repository contains a modular hurricane debris detection pipeline for
high-resolution UAV imagery. The project combines:

- **Florence-2** for open-vocabulary object detection
- **SAM2** for segmentation and mask refinement
- **Cross-dataset evaluation** across RescueNet, MSNet, and DesignSafe-CI
- **Structured export** for JSON and image-coordinate GeoJSON outputs
- **A Gradio demo** for local inference

For a deeper implementation/status summary, see
[`PROJECT_V3.md`](./PROJECT_V3.md).

## Repository Layout

```text
.
├── app.py                         # Gradio demo for local inference
├── main.py                        # Main CLI for download/train/evaluate/infer
├── hurricane_debris/              # Core package
├── scripts/run_experiments.py     # Baseline and ablation runner
├── scripts/experiment_matrix.json # Experiment matrix definition
├── requirements.txt               # Python dependencies
└── PROJECT_V3.md                  # Scope, alignment, and implementation notes
```

## Features

- Dataset download helpers with extraction and layout validation
- Training flows for Florence-2 and SAM2
- Cascaded inference from detection to segmentation
- Evaluation metrics including **mIoU**, **F1**, and **AP@[0.5:0.95]**
- Cross-dataset benchmarking from RescueNet to MSNet and DesignSafe-CI
- Experiment matrix support for baselines and ablations
- Deterministic run controls via `--seed`

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then install **SAM2** separately, because it is intentionally documented in
`requirements.txt` as an extra install step:

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Recommended Environment Notes

- Python environment with PyTorch support for your target device
- GPU acceleration is recommended for training and large-scale inference
- The CLI defaults to `--device auto`, which resolves to the best available
  backend at runtime

## Datasets

The project supports the following datasets:

- **RescueNet** - primary training and evaluation dataset
- **MSNet** - cross-dataset evaluation target
- **DesignSafe-CI** - cross-dataset evaluation target

Before training, download the required datasets. The built-in pipeline handles
download, extraction, and directory-layout validation.

```bash
# Download RescueNet (primary training/evaluation dataset)
python main.py --download --dataset rescuenet --dataset-dir ./datasets

# Download MSNet (cross-dataset evaluation)
python main.py --download --dataset msnet --dataset-dir ./datasets

# Download DesignSafe-CI (cross-dataset evaluation; requires free account access)
python main.py --download --dataset designsafe --dataset-dir ./datasets

# Download all datasets in one go
python main.py --download --dataset all --dataset-dir ./datasets

# Force re-download even if the data already exists
python main.py --download --dataset rescuenet --dataset-dir ./datasets --force-download

# Keep the downloaded archive after extraction
python main.py --download --dataset rescuenet --dataset-dir ./datasets --keep-archive
```

> **Note:** Some datasets, especially MSNet and DesignSafe-CI, may require
> registration or manual access approval. When automatic download is not
> available, the CLI prints instructions for manual acquisition.

## Quick Start

### Full pipeline

```bash
python main.py --full-pipeline --dataset rescuenet --dataset-dir ./datasets
```

### Train individual stages

```bash
python main.py --train-florence --dataset rescuenet --dataset-dir ./datasets
python main.py --train-sam2 --dataset rescuenet --dataset-dir ./datasets
```

### Evaluate

```bash
# Evaluate on one dataset
python main.py --evaluate --dataset rescuenet --dataset-dir ./datasets --metrics-dir ./outputs/metrics

# Cross-dataset evaluation using configured test datasets
python main.py --evaluate --dataset rescuenet --dataset-dir ./datasets --cross-dataset --metrics-dir ./outputs/metrics
```

### Run inference

```bash
python main.py --infer --image path/to/image.jpg
python main.py --infer --image path/to/image.jpg --output-json out.json
python main.py --infer --image path/to/image.jpg --output-json out.json --output-geojson out.geojson
```

## Model and Output Paths

The CLI uses these defaults unless you override them:

- `--florence-dir ./models/florence2_debris`
- `--sam2-dir ./models/sam2_debris`
- `--sam2-checkpoint ./checkpoints/sam2_hiera_large.pt`
- `--metrics-dir ./outputs/metrics`

Inference outputs can include:

- **Structured JSON** via `--output-json`
- **GeoJSON** via `--output-geojson`

> **GeoJSON scope:** current GeoJSON export is image-coordinate output. It is
> useful for downstream tooling, but it is not yet full GIS-georeferenced
> reprojection output.

## Baseline and Ablation Experiments

Run the experiment matrix and aggregate metrics:

```bash
python scripts/run_experiments.py --matrix scripts/experiment_matrix.json --output-dir ./outputs/experiments
```

This generates summary artifacts such as:

- `outputs/experiments/experiment_summary.json`
- `outputs/experiments/experiment_summary.csv`

## Gradio Demo

Launch the local web app:

```bash
python app.py --florence-dir ./models/florence2_debris --sam2-checkpoint ./models/sam2_debris/best_model.pth
```

Then open `http://127.0.0.1:7860` in your browser.

## Validation and Tests

The repository includes a pytest-based test suite under
`hurricane_debris/tests/`.

```bash
python -m pytest
python -m pytest --cov
python -m pytest hurricane_debris/tests/test_main_integration.py
```

If these commands fail with `No module named ...`, install the project
dependencies first with `pip install -r requirements.txt`.

## Troubleshooting

- **`python main.py --help` fails because a dependency is missing**  
  Install the dependencies from `requirements.txt` first.
- **SAM2 import errors**  
  Install SAM2 separately using the command shown in the setup section.
- **Checkpoint path errors during inference/evaluation**  
  Confirm that the Florence-2 and SAM2 output/checkpoint paths match the
  directories produced by your training runs.
- **Dataset download limitations**  
  Some datasets require external registration or manual download steps.

## Project Scope Notes

- The project is organized around a CLI-first workflow in `main.py`
- The Gradio app in `app.py` is intended for local demo usage
- `PROJECT_V3.md` documents implementation status, scope clarifications, and
  remaining research-oriented gaps
