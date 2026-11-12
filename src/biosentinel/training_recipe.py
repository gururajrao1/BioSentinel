"""Two-stage ResNet-18 fine-tuning recipe for small imbalanced wound data.

Stage 1: backbone frozen, train fresh head only.
Stage 2: unfreeze late stages (``layer4`` + ``fc``) at a lower LR.

Also centralises the §1 fixes: inverse-frequency class weights,
conservative augmentation, ImageNet normalisation, and
patient/group-aware splits (no group in both train and test).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["burn", "cuts_infection", "necrosis", "tissue_wound"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet18(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_stage1_head_only(model: nn.Module) -> nn.Module:
    """Freeze everything except the classification head."""
    from biosentinel.models import head_prefixes

    heads = head_prefixes(model)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(heads)
    return model


def set_stage2_late_unfreeze(model: nn.Module) -> nn.Module:
    """Freeze everything except late stages + head for low-LR tuning."""
    from biosentinel.models import late_prefixes

    late = late_prefixes(model)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(late)
    return model


def trainable_param_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def compute_class_weights(
    counts: list[int], max_ratio: float = 10.0
) -> torch.Tensor:
    """Inverse-frequency weights: total / count per class.

    Matches the notebook convention (``sum(counts) / c``) while guarding
    against empty classes (weight 1.0) and capping the max/min ratio at
    ``max_ratio`` so one rare class cannot dominate the loss.
    """
    if not counts or any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative with at least one class")
    total = float(sum(counts))
    if total == 0:
        raise ValueError("counts must sum to > 0")
    weights = [total / c if c > 0 else 1.0 for c in counts]
    floor = min(weights)
    cap = floor * max_ratio
    weights = [min(w, cap) for w in weights]
    return torch.tensor(weights, dtype=torch.float)


def make_train_transforms(size: int = 224) -> transforms.Compose:
    """Conservative augmentation: small rotation/shift, mild brightness."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_eval_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def group_train_test_split(
    groups: list[str], test_size: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split indices so no group appears on both sides (patient-level split)."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    indices = list(range(len(groups)))
    train_idx, test_idx = next(splitter.split(indices, groups=groups))
    return list(train_idx), list(test_idx)


def per_class_recall_from_confusion(cm) -> list[float]:
    """Recall per class from a confusion matrix (rows = true, cols = pred)."""
    recalls = []
    for i in range(len(cm)):
        denom = sum(cm[i])
        recalls.append(float(cm[i][i] / denom) if denom else 0.0)
    return recalls
