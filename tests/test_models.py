"""Tests for §3 architecture options and segment-then-derive-box."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.models import (
    EDGE_ARCHES,
    box_from_mask,
    build_classifier,
    soft_vote,
)


@pytest.mark.parametrize("arch", EDGE_ARCHES)
def test_build_classifier_forward_shape(arch):
    model = build_classifier(arch, num_classes=4, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 128, 128))
    assert out.shape == (1, 4)


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_classifier("dino_giant")


def test_soft_vote_averages_probabilities():
    a = [[0.6, 0.4], [0.3, 0.7]]
    b = [[0.4, 0.6], [0.3, 0.7]]
    assert soft_vote([a, b]) == [0, 1]


def test_box_from_mask_tight_box():
    mask = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    assert box_from_mask(mask) == (1, 1, 2, 2)


def test_box_from_mask_empty_is_none():
    assert box_from_mask([[0, 0], [0, 0]]) is None
    assert box_from_mask(torch.zeros(4, 4)) is None
