"""Uncertainty sampling: pick the most informative unlabeled cases next.

The data flywheel from the production plan — the model ranks the unlabeled
pool and humans label the top-k most uncertain images first.
"""

from __future__ import annotations


def uncertainty_scores(probs, method: str = "entropy") -> list[float]:
    """Higher score = more uncertain. Methods: entropy, least_confidence, margin."""
    import numpy as np

    probs = np.asarray(probs, dtype=float)
    if method == "entropy":
        logp = np.log(np.clip(probs, 1e-12, 1.0))
        return (-(probs * logp).sum(axis=1)).tolist()
    if method == "least_confidence":
        return (1.0 - probs.max(axis=1)).tolist()
    if method == "margin":
        part = np.partition(probs, -2, axis=1)[:, -2:]
        return (1.0 - abs(part[:, 1] - part[:, 0])).tolist()
    raise ValueError(f"unknown method {method!r}")


def rank_for_labeling(
    filenames: list[str], probs, top_k: int, method: str = "entropy"
) -> list[str]:
    """Filenames of the top-k most uncertain images, most uncertain first."""
    scores = uncertainty_scores(probs, method=method)
    ranked = sorted(zip(filenames, scores), key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in ranked[: max(0, int(top_k))]]
