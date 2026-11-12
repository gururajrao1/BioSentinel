"""Tests for dataset auditing."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.data_audit import (
    audit_dataset,
    count_images_per_class,
    find_bad_images,
    quarantine,
)


def _make(root: Path, split: str, counts: dict[str, int]):
    for cls, n in counts.items():
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.new("RGB", (16, 16), color=(i, 10, 10)).save(d / f"{i}.jpg")


def test_count_images_per_class(tmp_path):
    _make(tmp_path, "train", {"burn": 3, "necrosis": 1})
    assert count_images_per_class(str(tmp_path)) == {"burn": 3, "necrosis": 1}


def test_audit_flags_thin_classes(tmp_path):
    _make(tmp_path, "train", {"burn": 250, "necrosis": 12})
    report = audit_dataset(str(tmp_path), min_per_class=200)
    assert report["gaps"] == ["necrosis"]
    assert report["splits"]["train"]["burn"] == 250


def test_missing_split_returns_empty(tmp_path):
    assert count_images_per_class(str(tmp_path), split="val") == {}


def test_bad_images_found_and_quarantined(tmp_path):
    good = tmp_path / "train" / "burn"
    good.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(good / "ok.jpg")
    (good / "broken.jpg").write_bytes(b"not an image")
    (good / "empty.jpg").write_bytes(b"")
    bad = find_bad_images(str(tmp_path))
    assert sorted(reason for _, reason in bad) == ["corrupt", "empty"]
    moved = quarantine(bad, str(tmp_path / "quarantine"))
    assert len(moved) == 2
    assert count_images_per_class(str(tmp_path)) == {"burn": 1}
