"""Tests for uncertainty-sampling ranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.active_learning import rank_for_labeling, uncertainty_scores


def test_entropy_prefers_near_tie():
    scores = uncertainty_scores([[0.9, 0.1], [0.5, 0.5]])
    assert scores[1] > scores[0]


def test_all_methods_rank_tie_first():
    probs = [[0.9, 0.1], [0.5, 0.5], [0.7, 0.3]]
    names = ["a.jpg", "b.jpg", "c.jpg"]
    for method in ("entropy", "least_confidence", "margin"):
        assert rank_for_labeling(names, probs, top_k=1, method=method) == ["b.jpg"]
    assert rank_for_labeling(names, probs, top_k=2) == ["b.jpg", "c.jpg"]


def test_unknown_method_rejected():
    import pytest

    with pytest.raises(ValueError):
        uncertainty_scores([[0.5, 0.5]], method="nope")
