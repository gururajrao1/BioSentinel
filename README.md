# BioSentinel

**BioSentinel** is a machine learning pipeline for trauma and battlefield wound detection. Using limited medical imaging datasets, it classifies and visualizes wounds with around **74% accuracy** using a **ResNet-18** backbone. The project demonstrates end-to-end workflow: from data preprocessing to model training and evaluation, highlighting challenges of small datasets in medical AI.

## Features
- End-to-end ML pipeline for trauma wound detection
- Data preprocessing and augmentation
- ResNet-18 model for classification
- Visualization of wounds and predictions
- Demonstrates working with limited datasets

## Dataset
The model is trained on **limited medical imaging datasets** (not included in the repo due to size and privacy). You can provide your own datasets in the same format for experimentation. 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gururajrao1/BioSentinel.git
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

## Reproducible training (two-stage recipe)

```bash
python scripts/train.py --data dataset --arch resnet18
python scripts/compare.py --data dataset --arches resnet18 efficientnet_b4 mobilenet_v3_small
```

`--data` must be an ImageFolder layout with `train/`, `val/` (and optional
`test/`) subfolders per class. Metrics (accuracy, per-class recall/precision,
macro-F1, MCC, AUROC, confusion matrix, top confusions, temperature/ECE) land
in `outputs/` as `metrics_<arch>.json`, plus a `run.json` manifest (params,
metrics, data fingerprint) per run.

```bash
python scripts/predict.py --checkpoint outputs/best_resnet18.pth --arch resnet18 \
  --metrics-json outputs/metrics_resnet18.json --input path/to/images \
  --output-csv predictions.csv --threshold 0.85
```

`compare.py` also appends a soft-vote `ensemble(n=…)` row when given 2+ archs.
See [docs/labeling-guide.md](docs/labeling-guide.md) before labeling new data.

Notes

Accuracy is around 74% due to dataset scarcity.

Jupyter notebooks are included for visualizing workflow steps.

Do not commit large dataset files to this repo. Use .gitignore.

Dataset's Google drive link: https://drive.google.com/drive/folders/1Bg0JTFHt5JrQjfB3ledBjcY1T95-3QhO?usp=sharing
