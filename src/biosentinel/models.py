"""Architecture options (§3 of research-backed plan).

Edge: EfficientNet-B4 or nano-class models for classification+localisation.
Max accuracy: transformer/DINOv2-style or ensemble (kept as an interface —
heavy backbones stay optional so the repo never requires new weights).
Localisation for contiguous wounds: segment-then-derive-box.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

EDGE_ARCHES = (
    "convnext_tiny",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b4",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet18",
    "resnet34",
    "swin_t",
)


def head_prefixes(model) -> tuple[str, ...]:
    """Parameter-name prefixes of the classification head, per architecture."""
    kind = type(model).__name__
    if "ResNet" in kind:
        return ("fc.",)
    if "EfficientNet" in kind:
        return ("classifier.1.",)
    if "MobileNet" in kind:
        return ("classifier.3.",)
    if "DenseNet" in kind:
        return ("classifier.",)
    if "ConvNeXt" in kind:
        return ("classifier.2.",)
    if "SwinTransformer" in kind:
        return ("head.",)
    raise ValueError(f"no known head for model type {kind!r}")


def _last_block_prefix(model, holder: str = "features") -> str | None:
    indices = [
        int(parts[1])
        for n in (name for name, _ in model.named_parameters())
        if (parts := n.split(".")) and parts[0] == holder and parts[1].isdigit()
    ]
    return f"{holder}.{max(indices)}." if indices else None


def late_prefixes(model) -> tuple[str, ...]:
    """Parameter-name prefixes unfrozen in stage 2: head + last block."""
    kind = type(model).__name__
    if "ResNet" in kind:
        return ("layer4.", "fc.")
    if "DenseNet" in kind:
        return ("features.denseblock4.", "classifier.")
    last = _last_block_prefix(model)
    heads = head_prefixes(model)
    return ((last,) if last else ()) + heads


def build_classifier(
    arch: str, num_classes: int = 4, pretrained: bool = False
) -> nn.Module:
    """Build a classifier; pass pretrained=True for the §2.1 baseline.

    Default is random init so tests and offline use never download weights.
    """
    if arch in ("resnet18", "resnet34"):
        weights_enum = (
            models.ResNet18_Weights if arch == "resnet18" else models.ResNet34_Weights
        )
        builder = models.resnet18 if arch == "resnet18" else models.resnet34
        model = builder(weights=weights_enum.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b4"):
        weights_enum = {
            "efficientnet_b0": models.EfficientNet_B0_Weights,
            "efficientnet_b1": models.EfficientNet_B1_Weights,
            "efficientnet_b4": models.EfficientNet_B4_Weights,
        }[arch]
        builder = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b4": models.efficientnet_b4,
        }[arch]
        model = builder(weights=weights_enum.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if arch == "convnext_tiny":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model
    if arch == "swin_t":
        model = models.swin_t(
            weights=models.Swin_T_Weights.DEFAULT if pretrained else None
        )
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if arch in ("mobilenet_v3_small", "mobilenet_v3_large"):
        weights_enum = (
            models.MobileNet_V3_Small_Weights
            if arch == "mobilenet_v3_small"
            else models.MobileNet_V3_Large_Weights
        )
        builder = (
            models.mobilenet_v3_small
            if arch == "mobilenet_v3_small"
            else models.mobilenet_v3_large
        )
        model = builder(weights=weights_enum.DEFAULT if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"unknown arch {arch!r}; expected one of {EDGE_ARCHES}")


def soft_vote(probas_list) -> list[int]:
    """Heterogeneous-ensemble vote: average probabilities, pick highest."""
    import numpy as np

    mean = np.mean(np.asarray(probas_list, dtype=float), axis=0)
    return mean.argmax(axis=1).tolist()


def box_from_mask(mask) -> tuple[int, int, int, int] | None:
    """Tight (x, y, w, h) box around nonzero mask pixels; None if empty.

    Accepts nested sequences or tensors with shape (H, W).
    """
    try:
        import torch

        if isinstance(mask, torch.Tensor):
            m = mask.detach().cpu()
            ys, xs = (m > 0).nonzero(as_tuple=True)
            if xs.numel() == 0:
                return None
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    except ImportError:
        pass
    rows = [i for i, r in enumerate(mask) for v in r if v]
    cols = [j for r in mask for j, v in enumerate(r) if v]
    if not rows or not cols:
        return None
    x0, x1 = min(cols), max(cols)
    y0, y1 = min(rows), max(rows)
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
