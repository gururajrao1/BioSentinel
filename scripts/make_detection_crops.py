"""Crop YOLO detection boxes into a classification folder layout.

Keeps the source train/valid/test splits so nothing leaks across them.
Only mapped class ids are kept; everything else is skipped.

Example:
    python scripts/make_detection_crops.py --data <yolo-root> \\
        --out data/crops --map 0:burn,1:burn,2:burn --pad 0.1
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_class_map(spec: str) -> dict[int, str]:
    """Parse '0:burn,1:burn,2:burn' into {0: 'burn', ...}."""
    mapping = {}
    for item in spec.split(","):
        key, _, value = item.partition(":")
        mapping[int(key.strip())] = value.strip()
    return mapping


def crop_boxes(
    data_dir: str,
    out_dir: str,
    class_map: dict[int, str],
    pad: float = 0.1,
    min_size: int = 64,
) -> dict[str, dict[str, int]]:
    """Crop mapped boxes; returns {split: {class: count}}."""
    from biosentinel.data_audit import IMAGE_EXTS

    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(data_dir, split, "images")
        lbl_dir = os.path.join(data_dir, split, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.isfile(label_path):
                continue
            try:
                img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            with open(label_path) as f:
                lines = [ln.split() for ln in f if ln.strip()]
            for i, parts in enumerate(lines):
                cls = int(float(parts[0]))
                if cls not in class_map:
                    continue
                _, xc, yc, bw, bh = (float(v) for v in parts[:5])
                x0 = max(0, int((xc - bw / 2 - pad * bw) * w))
                y0 = max(0, int((yc - bh / 2 - pad * bh) * h))
                x1 = min(w, int((xc + bw / 2 + pad * bw) * w))
                y1 = min(h, int((yc + bh / 2 + pad * bh) * h))
                if x1 - x0 < min_size or y1 - y0 < min_size:
                    continue
                dest = os.path.join(out_dir, split, class_map[cls])
                os.makedirs(dest, exist_ok=True)
                name = f"{stem}_box{i}.jpg"
                img.crop((x0, y0, x1, y1)).save(os.path.join(dest, name), quality=92)
                counts.setdefault(split, {}).setdefault(class_map[cls], 0)
                counts[split][class_map[cls]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop YOLO boxes for classification.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--pad", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=64)
    args = parser.parse_args()
    counts = crop_boxes(args.data, args.out, parse_class_map(args.map), args.pad, args.min_size)
    for split in ("train", "valid", "test"):
        print(split, counts.get(split, {}))


if __name__ == "__main__":
    main()
