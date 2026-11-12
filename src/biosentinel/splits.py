"""Patient/lesion-aware splitting (§1.1 of production plan).

Same patient (or lesion / source case) must never appear on both sides of a
split. Use the locked holdout once, and K-fold inside the development pool.
"""

from __future__ import annotations

import numpy as np


def stratified_group_kfold_indices(
    labels: list, groups: list, n_splits: int = 5, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    """Group-aware stratified K-fold indices with no group on both sides."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return [
        (list(tr), list(te))
        for tr, te in splitter.split(
            np.zeros(len(labels)), np.asarray(labels), np.asarray(groups)
        )
    ]


def locked_holdout_indices(
    groups: list, test_size: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Single group-aware holdout. Fix the seed and touch the set once."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    idx = np.arange(len(groups))
    train_idx, test_idx = next(splitter.split(idx, groups=np.asarray(groups)))
    return list(train_idx), list(test_idx)
