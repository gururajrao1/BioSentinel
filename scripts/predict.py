"""Batch inference with temperature scaling and an abstain-to-human queue.

Reads the temperature and class list from a training metrics file when given,
writes one row per image, and reports coverage (1 - abstention rate).

Example:
    python scripts/predict.py --checkpoint outputs/best_resnet18.pth \\
        --arch resnet18 --metrics-json outputs/metrics_resnet18.json \\
        --input path/to/images --output-csv predictions.csv --threshold 0.85
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.edge import serving_transforms
from biosentinel.models import build_classifier

ABSTAIN = "ABSTAIN"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(IMAGE_EXTS)
    )


def load_model(arch: str, checkpoint: str, num_classes: int, device: str):
    model = build_classifier(arch, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def predict_batch(
    model,
    image_paths: list[str],
    classes: list[str],
    img_size: int = 224,
    temperature: float = 1.0,
    threshold: float = 0.0,
    device: str = "cpu",
) -> list[dict]:
    """Predict each image; abstain when confidence is below threshold."""
    transform = serving_transforms(img_size)
    temperature = max(float(temperature), 1e-3)
    rows = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            logits = model(transform(img).unsqueeze(0).to(device))
            probs = torch.softmax(logits / temperature, dim=1)[0].tolist()
            confidence = max(probs)
            predicted = classes[probs.index(confidence)]
            abstained = confidence < threshold
            rows.append(
                {
                    "filename": os.path.basename(path),
                    "predicted": ABSTAIN if abstained else predicted,
                    "confidence": round(confidence, 4),
                    "abstained": abstained,
                }
            )
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "predicted", "confidence", "abstained"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="BioSentinel batch inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output-csv", default="predictions.csv")
    args = parser.parse_args()

    classes = list(args.classes) if args.classes else None
    temperature = args.temperature
    if args.metrics_json:
        with open(args.metrics_json) as f:
            saved = json.load(f)
        classes = classes or saved["classes"]
        temperature = saved.get("calibration", {}).get("temperature", temperature)
    if not classes:
        parser.error("provide --classes or --metrics-json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.arch, args.checkpoint, len(classes), device)
    rows = predict_batch(
        model,
        list_images(args.input),
        classes,
        img_size=args.img_size,
        temperature=temperature,
        threshold=args.threshold,
        device=device,
    )
    write_csv(rows, args.output_csv)
    abstained = sum(1 for r in rows if r["abstained"])
    print(f"{len(rows)} images, {abstained} abstained, coverage={1 - abstained / max(len(rows), 1):.2f}")


if __name__ == "__main__":
    main()
