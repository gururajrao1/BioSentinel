"""Temperature scaling + abstain-to-human routing (§2.5 of production plan).

Fit the temperature on held-out development data only, then operate at an
explicit threshold: below it the case abstains to a human and coverage drops.
"""

from __future__ import annotations


def fit_temperature(logits, labels, max_iter: int = 50) -> float:
    """Scale factor T minimizing NLL on held-out logits. T > 1 cools."""
    import torch
    import torch.nn as nn

    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        loss = criterion(z / temperature.clamp_min(1e-3), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-3).item())


def apply_temperature(logits, temperature: float):
    import torch
    import torch.nn.functional as F

    z = torch.tensor(logits, dtype=torch.float32)
    return F.softmax(z / max(float(temperature), 1e-3), dim=1).tolist()


def ece_score(probs, labels, n_bins: int = 15) -> float:
    """Expected calibration error in [0, 1]."""
    import numpy as np

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(labels)) * abs(
            (pred[mask] == labels[mask]).mean() - conf[mask].mean()
        )
    return float(ece)


def route_with_abstention(probs, threshold: float = 0.85) -> tuple[list[int], float]:
    """Predict argmax, or -1 (abstain to human) when confidence is low."""
    import numpy as np

    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1).tolist()
    routed = [p if c >= threshold else -1 for p, c in zip(preds, conf)]
    coverage = float(sum(1 for r in routed if r != -1) / len(routed)) if routed else 0.0
    return routed, coverage
