"""Tests for the §2 classical-first augmentation policy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.augmentation import (
    cap_synthetic_count,
    classical_train_transforms,
    keep_synthetic,
)


def test_classical_transforms_emit_224_tensor():
    from PIL import Image

    img = Image.new("RGB", (300, 200), color=(90, 40, 160))
    out = classical_train_transforms()(img)
    assert out.shape == (3, 224, 224)


def test_synthetic_cap():
    assert cap_synthetic_count(1000) == 400
    assert cap_synthetic_count(50) == 50
    assert cap_synthetic_count(0) == 0


def test_synthetic_gate_requires_strict_gain():
    assert keep_synthetic(0.01) is True
    assert keep_synthetic(0.0) is False
    assert keep_synthetic(-0.02) is False
