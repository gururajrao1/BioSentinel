"""Tests for K-fold layout building (no training)."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from crossval import collect_samples, make_fold_layout


def _make(root: Path):
    for split in ("train", "val"):
        for cls, n in (("burn", 6), ("necrosis", 6)):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.new("RGB", (16, 16)).save(d / f"{split}_{i}.jpg")


def test_folds_cover_all_samples_without_overlap(tmp_path):
    _make(tmp_path / "data")
    paths, labels, names = collect_samples(str(tmp_path / "data"))
    assert len(paths) == 24 and names == ["burn", "necrosis"]
    folds = make_fold_layout(paths, labels, names, 2, 0, str(tmp_path / "cv"))
    for fold in folds:
        train = list((Path(fold) / "train" / "burn").glob("*.jpg"))
        val = list((Path(fold) / "val" / "burn").glob("*.jpg"))
        assert len(train) + len(val) == 12
        assert {p.name for p in train}.isdisjoint({p.name for p in val})
