"""Dataset auditing (§1.3/§1.4 of production plan).

Count images per class and split, and flag thin classes before training.
Target: a few hundred labeled images per class with a pretrained backbone.
"""

from __future__ import annotations

import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def count_images_per_class(data_dir: str, split: str = "train") -> dict[str, int]:
    """Map class name to image count for one split of an ImageFolder layout."""
    base = os.path.join(data_dir, split)
    counts: dict[str, int] = {}
    if not os.path.isdir(base):
        return counts
    for cls in sorted(os.listdir(base)):
        cls_dir = os.path.join(base, cls)
        if not os.path.isdir(cls_dir):
            continue
        counts[cls] = sum(
            1 for f in os.listdir(cls_dir) if f.lower().endswith(IMAGE_EXTS)
        )
    return counts


def find_bad_images(data_dir: str, split: str = "train") -> list[tuple[str, str]]:
    """Corrupt or zero-byte images as (path, reason) pairs."""
    from PIL import Image

    bad: list[tuple[str, str]] = []
    base = os.path.join(data_dir, split)
    if not os.path.isdir(base):
        return bad
    for cls in sorted(os.listdir(base)):
        cls_dir = os.path.join(base, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if not f.lower().endswith(IMAGE_EXTS):
                continue
            path = os.path.join(cls_dir, f)
            if os.path.getsize(path) == 0:
                bad.append((path, "empty"))
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                bad.append((path, "corrupt"))
    return bad


def quarantine(bad: list[tuple[str, str]], quarantine_dir: str) -> list[str]:
    """Move bad files aside (prefixed by class) instead of deleting them."""
    import shutil

    moved = []
    os.makedirs(quarantine_dir, exist_ok=True)
    for path, _ in bad:
        parent = os.path.basename(os.path.dirname(path))
        dest = os.path.join(quarantine_dir, f"{parent}_{os.path.basename(path)}")
        shutil.move(path, dest)
        moved.append(dest)
    return moved


def audit_dataset(data_dir: str, min_per_class: int = 200) -> dict:
    """Per-split counts plus the list of classes below target."""
    splits = {
        split: count_images_per_class(data_dir, split)
        for split in ("train", "val", "test")
    }
    train_counts = splits["train"]
    gaps = sorted(
        cls for cls, n in train_counts.items() if n < min_per_class
    )
    return {"splits": splits, "min_per_class": min_per_class, "gaps": gaps}
