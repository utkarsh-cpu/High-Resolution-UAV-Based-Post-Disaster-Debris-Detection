# CV-Project

Hurricane debris detection project with training, evaluation, and inference pipelines.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Examples:

```bash
python main.py --full-pipeline
python main.py --train-florence --dataset rescuenet
python main.py --train-sam2 --dataset rescuenet
python main.py --evaluate --dataset msnet
python main.py --infer --image path/to/image.jpg
```
