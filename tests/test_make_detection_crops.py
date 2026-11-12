"""Tests for YOLO box cropping."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_detection_crops import crop_boxes, parse_class_map


def _yolo(root: Path):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    Image.new("RGB", (200, 200), color=(200, 50, 50)).save(root / "train" / "images" / "a.jpg")
    # box 0 (mapped), box 3 (unmapped), tiny box 1 (filtered)
    (root / "train" / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.6 0.6\n3 0.5 0.5 0.6 0.6\n1 0.1 0.1 0.05 0.05\n"
    )


def test_parse_class_map():
    assert parse_class_map("0:burn,1:burn,2:burn") == {0: "burn", 1: "burn", 2: "burn"}


def test_crop_boxes_keeps_only_mapped_big_enough(tmp_path):
    _yolo(tmp_path / "yolo")
    counts = crop_boxes(
        str(tmp_path / "yolo"),
        str(tmp_path / "out"),
        {0: "burn", 1: "burn", 2: "burn"},
        min_size=64,
    )
    assert counts == {"train": {"burn": 1}}
    crops = list((tmp_path / "out" / "train" / "burn").glob("*.jpg"))
    assert len(crops) == 1
    with Image.open(crops[0]) as img:
        assert img.size[0] >= 64 and img.size[1] >= 64
