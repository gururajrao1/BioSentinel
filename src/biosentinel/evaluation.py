"""Clinical evaluation (§4 of research-backed plan).

Lock a patient/group holdout, run grouped CV inside the training pool,
and report discrimination (AUROC) with sensitivity/specificity —
never accuracy alone. OOD/device-shift gap is a first-class number.
"""

from __future__ import annotations


def per_class_sensitivity_specificity(cm) -> dict:
    """Sensitivity (recall) and one-vs-rest specificity per class."""
    out: dict = {"sensitivity": [], "specificity": []}
    n = len(cm)
    for i in range(n):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(n)) - tp
        tn = sum(sum(row) for row in cm) - tp - fn - fp
        out["sensitivity"].append(tp / (tp + fn) if tp + fn else 0.0)
        out["specificity"].append(tn / (tn + fp) if tn + fp else 0.0)
    return out


def multiclass_auroc(y_true, y_score) -> float:
    """One-vs-rest macro AUROC; returns NaN when a class has one label."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    try:
        scores = np.asarray(y_score)
        if scores.ndim == 2 and scores.shape[1] == 2:
            return float(roc_auc_score(y_true, scores[:, 1]))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except ValueError:
        return float("nan")


def ood_gap(id_score: float, ood_score: float) -> float:
    """Positive when the OOD/device-shift split drops vs in-domain."""
    return float(id_score) - float(ood_score)


def extended_metrics(y_true, y_pred) -> dict:
    """Per-class precision/recall, macro-F1 and MCC for imbalanced classes."""
    from sklearn.metrics import (
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = float("nan")
    return {
        "per_class_precision": [float(v) for v in precision],
        "per_class_recall": [float(v) for v in recall],
        "per_class_f1": [float(v) for v in f1],
        "macro_f1": float(sum(f1) / len(f1)) if len(f1) else 0.0,
        "mcc": mcc,
    }


def most_confused_pairs(cm, classes, top_k: int = 3) -> list[tuple]:
    """Top (true, predicted, count) off-diagonal confusions to review."""
    pairs = [
        (classes[i], classes[j], int(cm[i][j]))
        for i in range(len(cm))
        for j in range(len(cm))
        if i != j and cm[i][j] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top_k]


def grouped_cv_splits(
    groups: list[str], n_splits: int = 5, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    """Patient/group-aware CV splits with no group on both sides."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    # StratifiedGroupKFold needs a label per sample; use group ids as a
    # stable dummy stratification so splits stay grouped and deterministic.
    y = np.array(groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (list(tr), list(te)) for tr, te in splitter.split(np.zeros(len(groups)), y, groups)
    ]
