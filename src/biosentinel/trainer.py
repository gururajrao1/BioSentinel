"""Shared two-stage training loop implementing the §1 recipe.

Stage 1 trains the fresh head with the backbone frozen; stage 2 unfreezes
late stages + head at a lower LR. Reports per-class precision/recall/F1,
macro-F1, MCC and AUROC (never accuracy alone), fits temperature scaling on
validation, and writes a per-run manifest (params + metrics + data hash).
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from biosentinel import calibration as cal
from biosentinel import evaluation as ev
from biosentinel import runs
from biosentinel.models import build_classifier
from biosentinel.training_recipe import (
    compute_class_weights,
    make_eval_transforms,
    make_train_transforms,
    set_stage1_head_only,
    set_stage2_late_unfreeze,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FocalLoss(nn.Module):
    """Cross-entropy down-weighting easy examples; helps rare classes."""

    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits, targets):
        import torch.nn.functional as F

        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            reduction="none",
        )
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


def build_criterion(loss_name: str, weights, label_smoothing: float) -> nn.Module:
    if loss_name == "focal":
        return FocalLoss(weight=weights, gamma=2.0)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    raise ValueError(f"unknown loss {loss_name!r}; choose 'ce' or 'focal'")


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay over the whole run."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_steps = max(0, min(int(warmup_steps), int(total_steps) - 1))
    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, int(total_steps)))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, int(total_steps) - warmup_steps)
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def _amp_dtype(device: str):
    """Mixed-precision dtype per device, or None for plain fp32."""
    if device == "cuda":
        return torch.float16
    try:
        torch.empty(1, dtype=torch.bfloat16).to("cpu")
        return torch.bfloat16
    except Exception:
        return None


def _run_epochs(
    model, loader, criterion, optimizer, device, scheduler=None, use_amp: bool = False
):
    model.train()
    amp_dtype = _amp_dtype(device) if use_amp else None
    scaler = torch.amp.GradScaler("cuda") if use_amp and device == "cuda" else None
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if amp_dtype is None:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        elif scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss, correct / total if total else 0.0


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    all_labels, all_probs, all_logits = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.extend(labels.tolist())
    return (
        all_labels,
        torch.cat(all_logits).tolist() if all_logits else [],
        torch.cat(all_probs).tolist() if all_probs else [],
    )


def _evaluate(model, loader, device, classes):
    labels, _, probs = _collect(model, loader, device)
    preds = (np.asarray(probs).argmax(1).tolist()) if probs else []
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes)))).tolist()
    stats = ev.per_class_sensitivity_specificity(cm)
    summary = {
        "accuracy": float(np.mean([p == t for p, t in zip(preds, labels)]))
        if labels
        else 0.0,
        "per_class_recall": stats["sensitivity"],
        "per_class_specificity": stats["specificity"],
        "auroc": ev.multiclass_auroc(labels, probs),
        "confusion_matrix": cm,
    }
    summary.update(ev.extended_metrics(labels, preds))
    summary["top_confusions"] = [
        {"true": t, "predicted": p, "count": c}
        for t, p, c in ev.most_confused_pairs(cm, classes)
    ]
    return summary


def train_classifier(
    data_dir: str,
    arch: str = "resnet18",
    img_size: int = 224,
    batch_size: int = 32,
    epochs_stage1: int = 3,
    epochs_stage2: int = 3,
    lr_head: float = 1e-3,
    lr_late: float = 1e-4,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.01,
    loss_name: str = "ce",
    label_smoothing: float = 0.1,
    warmup_epochs: int = 1,
    calibrate: bool = True,
    pretrained: bool = True,
    use_amp: bool = True,
    patience: int | None = 3,
    seed: int = 42,
    device: str | None = None,
    output_dir: str = "outputs",
) -> dict:
    """Train one architecture with the two-stage recipe; save best + metrics."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=make_train_transforms(img_size)
    )
    val_ds = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=make_eval_transforms(img_size)
    )
    test_dir = os.path.join(data_dir, "test")
    test_ds = (
        datasets.ImageFolder(test_dir, transform=make_eval_transforms(img_size))
        if os.path.isdir(test_dir)
        else None
    )
    classes = train_ds.classes
    counts = [sum(1 for _, c in train_ds.samples if c == i) for i in range(len(classes))]
    weights = compute_class_weights(counts).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_classifier(
        arch, num_classes=len(classes), pretrained=pretrained
    ).to(device)
    criterion = build_criterion(loss_name, weights, label_smoothing)
    opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[optimizer_name]

    # One optimizer over all params (frozen params get no grad, so the
    # optimizer skips them) and one warmup+cosine schedule over both stages.
    # Stage 1 trains the head; stage 2 continues on late layers + head.
    set_stage1_head_only(model)
    total_steps = (epochs_stage1 + epochs_stage2) * max(1, len(train_loader))
    optimizer = opt_cls(
        model.parameters(), lr=lr_head, weight_decay=weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_epochs * max(1, len(train_loader)),
        total_steps=total_steps,
    )
    best_f1, best_state, epochs_run = -1.0, None, 0
    stale = 0
    plan = [(set_stage1_head_only, epochs_stage1, lr_head),
            (set_stage2_late_unfreeze, epochs_stage2, lr_late)]
    for freeze_fn, epochs, lr in plan:
        freeze_fn(model)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(epochs):
            _run_epochs(
                model, train_loader, criterion, optimizer, device, scheduler,
                use_amp=use_amp,
            )
            epochs_run += 1
            val_f1 = _evaluate(model, val_loader, device, classes)["macro_f1"]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if patience is not None and stale >= patience:
                break
        if patience is not None and stale >= patience:
            break
    metrics_trained_epochs = epochs_run

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    val_labels, val_logits, val_probs = _collect(model, val_loader, device)
    metrics: dict = {
        "arch": arch,
        "classes": classes,
        "val": _evaluate(model, val_loader, device, classes),
    }
    if calibrate and val_logits:
        temperature = cal.fit_temperature(val_logits, val_labels)
        metrics["calibration"] = {
            "temperature": temperature,
            "ece_before": cal.ece_score(val_probs, val_labels),
            "ece_after": cal.ece_score(
                cal.apply_temperature(val_logits, temperature), val_labels
            ),
        }
    if test_ds is not None:
        metrics["test"] = _evaluate(
            model, DataLoader(test_ds, batch_size=batch_size), device, classes
        )

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"best_{arch}.pth"))
    with open(os.path.join(output_dir, f"metrics_{arch}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    runs.log_run(
        output_dir,
        params={
            "arch": arch,
            "img_size": img_size,
            "batch_size": batch_size,
            "epochs_stage1": epochs_stage1,
            "epochs_stage2": epochs_stage2,
            "lr_head": lr_head,
            "lr_late": lr_late,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "loss": loss_name,
            "label_smoothing": label_smoothing,
            "warmup_epochs": warmup_epochs,
            "pretrained": pretrained,
            "use_amp": use_amp,
            "patience": patience,
            "epochs_run": metrics_trained_epochs,
            "seed": seed,
        },
        metrics={
            "val_accuracy": metrics["val"]["accuracy"],
            "val_auroc": metrics["val"]["auroc"],
            "val_macro_f1": metrics["val"]["macro_f1"],
            "val_mcc": metrics["val"]["mcc"],
        },
        data_dir=data_dir,
    )
    return metrics
