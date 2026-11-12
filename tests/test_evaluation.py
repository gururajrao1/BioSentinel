"""Tests for §4 clinical evaluation helpers."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.evaluation import (
    extended_metrics,
    grouped_cv_splits,
    most_confused_pairs,
    multiclass_auroc,
    ood_gap,
    per_class_sensitivity_specificity,
)


def test_sensitivity_specificity_spots_weak_class():
    cm = [[8, 2], [5, 5]]
    m = per_class_sensitivity_specificity(cm)
    assert m["sensitivity"] == [0.8, 0.5]
    assert m["specificity"] == [0.5, 0.8]


def test_auroc_perfect_is_one():
    y_true = [0, 0, 1, 1]
    y_score = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    assert multiclass_auroc(y_true, y_score) == 1.0


def test_auroc_single_label_is_nan():
    assert math.isnan(multiclass_auroc([0, 0], [[0.5, 0.5], [0.5, 0.5]]))


def test_ood_gap_sign():
    assert ood_gap(0.85, 0.70) == pytest.approx(0.15)
    assert ood_gap(0.70, 0.85) < 0


def test_grouped_cv_has_no_group_leak():
    groups = [f"p{i // 2}" for i in range(20)]
    for tr, te in grouped_cv_splits(groups, n_splits=2, seed=0):
        assert {groups[i] for i in tr}.isdisjoint({groups[i] for i in te})


def test_extended_metrics_reports_imbalance_numbers():
    m = extended_metrics([0, 0, 0, 1, 1], [0, 0, 1, 1, 1])
    assert m["per_class_precision"][0] == 1.0
    assert m["per_class_recall"] == [2 / 3, 1.0]
    assert 0.0 < m["macro_f1"] < 1.0
    assert -1.0 <= m["mcc"] <= 1.0


def test_most_confused_pairs_ranks_off_diagonal():
    cm = [[8, 2], [5, 5]]
    assert most_confused_pairs(cm, ["a", "b"]) == [("b", "a", 5), ("a", "b", 2)]
