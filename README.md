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

## Project status & resume guide (Nov 2026)

TL;DR: best system is focal-loss ConvNeXt-Tiny at burn-0.92/rest-0.85
thresholds with TTA — 0.996 on the 235 test images it decides, 11 abstained
for human review. 3-fold CV macro-F1 is 0.973 ± 0.025, so the headline is
stable. The open wound is burn bias out-of-distribution.

### Data inventory (all under `data/`, git-ignored; source: Drive link above)

- `data/drive/BioSentinel.zip` — the 985 MB Drive download, untouched.
- `data/drive/ex/content/BioSentinel/dataset/` — original ImageFolder set:
  train 370 (burn 91, cuts_infection 56, necrosis 174, tissue_wound 49),
  val 28, test 22.
- `data/dataset-big/` — original + 3,076 burn crops from Skin-Burn-2
  detection boxes (`scripts/make_detection_crops.py`, splits preserved).
  Train 870 (burn 591), val 462, test 246.
- `data/dataset-cap174/` — same but burn train capped at 174 (train 453).
- `data/fuseg/` — 1,210 foot-ulcer photos (different source, no usable
  labels) used as an unlabeled OOD probe.
- `data/notburn_peek/` — sample of Skin-Burn's "Not burn" class: inspected,
  rejected (healthy stock photos, not wounds).
- Skipped deliberately: WoundCare `W002/W008` codes (unmappable, no codebook),
  `merged_dataset` (verified byte-for-byte to be Skin-Burn-2 duplicates),
  FUSeg masks (segmentation task, not our classes).

### What was tried, in order, and what it gave

1. Research-backed training recipe (`src/biosentinel/`, `scripts/train.py`,
   `compare.py`, `predict.py`; 60+ pytest tests). Two-stage fine-tune,
   class weights, group splits, calibration + abstain queue.
2. From-scratch 4-arch run: ResNet-18 0.714/0.818; EfficientNets/MobileNet
   collapsed (~0.15) — diagnosed as missing pretrained weights, fixed by
   making ImageNet init the default (`--no-pretrained` opts out).
3. Pretrained 4-arch run: ResNet-18 0.893/0.909, EfficientNet-B0 0.857/0.909,
   MobileNetV3-Small 0.857/0.864. Ensemble 0.929 val.
4. +500 burn crops: enabled by `make_detection_crops.py`.
5. GPU (RTX 3050, torch cu126) 10-arch run: Swin-T 0.907/0.943 best single,
   ConvNeXt-Tiny 0.997 AUROC, MobileNetV3-Small the edge pick.
6. Focal loss: ConvNeXt-Tiny 0.981/0.980, F1 0.839; 4-model ensemble
   0.976 val, F1 0.842.
7. Balanced sampler: no gain over focal alone (bias is in the features,
   not the batches). Per-class burn bar 0.92 + TTA: test 238 decided at
   0.983; FUSeg false burns 441 → 426, abstention 37% → 40%.
8. Burn-capped retrain (train burn 174): ConvNeXt 0.965/0.963, F1 0.770 —
   slightly worse in-distribution, slightly less biased OOD. Bias bends,
   doesn't break.

### How to reproduce

```bash
pip install -r requirements.txt              # CUDA build needs torch from the cu126 index (see below)
python scripts/compare.py --data data/dataset-big --arches convnext_tiny swin_t \
  --epochs-stage1 6 --epochs-stage2 10 --batch-size 16 --loss focal   # needs GPU
python scripts/predict.py --checkpoint outputs/<run>/best_convnext_tiny.pth \
  --arch convnext_tiny --metrics-json outputs/<run>/metrics_convnext_tiny.json \
  --input <images> --output-csv preds.csv --threshold 0.85 \
  --per-class-thresholds 'burn:0.92' --tta
python scripts/crossval.py --data data/dataset-big --arch convnext_tiny --folds 3 --loss focal
python -m pytest tests/ -q
```

CUDA note: the default `pip install torch` resolved to a CPU-only build
here; the working setup is `torch==2.14.1+cu126` from
`https://download.pytorch.org/whl/cu126` on the RTX 3050 (4 GB, hence
`--batch-size 16`).

### Open next steps

- Minority-class pixels: label the 1,210 foot ulcers, or find real
  non-burn wound sources. This beats any training trick left.
- Carried, not started: EfficientNet-B4/ConvNeXt/Swin heavies fully tuned,
  QAT + ONNX/XNNPACK on-device numbers, patient-level splits (needs IDs).
- `outputs/` (checkpoints, metrics, results.md per run) and `data/` are
  git-ignored by design; `run.json` in each output dir records the exact
  params + data fingerprint for every number above.
