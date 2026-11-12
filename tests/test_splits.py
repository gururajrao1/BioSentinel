"""Tests for patient/lesion-aware splitting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.splits import (
    locked_holdout_indices,
    stratified_group_kfold_indices,
)


def _toy():
    labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    return labels, groups


def test_stratified_group_kfold_has_no_group_leak():
    labels, groups = _toy()
    folds = stratified_group_kfold_indices(labels, groups, n_splits=2, seed=0)
    assert len(folds) == 2
    for tr, te in folds:
        assert {groups[i] for i in tr}.isdisjoint({groups[i] for i in te})
        assert len(tr) + len(te) == len(groups)


def test_locked_holdout_has_no_group_leak():
    _, groups = _toy()
    tr, te = locked_holdout_indices(groups, test_size=0.2, seed=0)
    assert {groups[i] for i in tr}.isdisjoint({groups[i] for i in te})
    assert len(tr) + len(te) == len(groups)
    # Same seed gives the same locked split.
    assert locked_holdout_indices(groups, test_size=0.2, seed=0) == (tr, te)
