"""Tests for temperature scaling and abstention routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.calibration import (
    apply_temperature,
    ece_score,
    fit_temperature,
    route_with_abstention,
)


def test_temperature_cools_overconfident_mistakes():
    logits = [[3.0, 0.0]] * 8 + [[0.0, 3.0]] * 2 + [[3.0, 0.0]] * 2
    labels = [0] * 8 + [1] * 2 + [1] * 2
    t = fit_temperature(logits, labels)
    assert t > 1.0


def test_ece_perfect_is_zero():
    probs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    assert ece_score(probs, [0, 0, 1, 1], n_bins=2) == 0.0


def test_abstention_threshold_controls_coverage():
    probs = [[0.9, 0.1], [0.55, 0.45], [0.05, 0.95]]
    routed, coverage = route_with_abstention(probs, threshold=0.85)
    assert routed == [0, -1, 1]
    assert coverage == 2 / 3
