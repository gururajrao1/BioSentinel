"""Tests for batch inference."""

import csv
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from biosentinel.models import build_classifier
from predict import ABSTAIN, list_images, predict_batch, write_csv


def _images(tmp_path: Path, n: int = 2) -> list[str]:
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (48, 48), color=(i * 30, 90, 40)).save(p)
        paths.append(str(p))
    return paths


def _checkpoint(tmp_path: Path) -> str:
    model = build_classifier("resnet18", num_classes=2)
    path = str(tmp_path / "ckpt.pth")
    torch.save(model.state_dict(), path)
    return path


def test_predict_batch_writes_reviewable_rows(tmp_path):
    from predict import load_model

    paths = _images(tmp_path)
    model = load_model("resnet18", _checkpoint(tmp_path), 2, "cpu")
    rows = predict_batch(model, paths, ["burn", "necrosis"], img_size=32)
    assert [r["filename"] for r in rows] == ["0.jpg", "1.jpg"]
    assert all(r["predicted"] in ("burn", "necrosis") for r in rows)
    out = str(tmp_path / "preds.csv")
    write_csv(rows, out)
    with open(out) as f:
        assert len(list(csv.DictReader(f))) == 2


def test_high_threshold_abstains_everything(tmp_path):
    from predict import load_model

    paths = _images(tmp_path)
    model = load_model("resnet18", _checkpoint(tmp_path), 2, "cpu")
    rows = predict_batch(
        model, paths, ["burn", "necrosis"], img_size=32, threshold=1.01
    )
    assert all(r["predicted"] == ABSTAIN and r["abstained"] for r in rows)


def test_tta_is_deterministic_and_valid(tmp_path):
    from predict import load_model

    paths = _images(tmp_path)
    model = load_model("resnet18", _checkpoint(tmp_path), 2, "cpu")
    rows_a = predict_batch(model, paths, ["burn", "necrosis"], img_size=32, tta=True)
    rows_b = predict_batch(model, paths, ["burn", "necrosis"], img_size=32, tta=True)
    assert rows_a == rows_b
    assert all(r["predicted"] in ("burn", "necrosis") for r in rows_a)


def test_list_images_accepts_file_or_dir(tmp_path):
    paths = _images(tmp_path, n=2)
    assert list_images(paths[0]) == [paths[0]]
    assert list_images(str(tmp_path)) == sorted(paths)
