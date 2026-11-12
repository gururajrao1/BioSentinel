"""Classical-first augmentation (§2 of research-backed plan).

Validated wound policy: on-the-fly rotation 0-30deg + random brightness.
Geometric augmentation added up to +11% F1 for low-F1 classes D/P/V.
Synthetic images (GAN/diffusion/3D) are only kept when they improve a
real held-out split; detection experiments cap synthetic at ~400.
"""

from __future__ import annotations

SYNTHETIC_CAP_DETECTION = 400
ROTATION_DEGREES = 30
BRIGHTNESS = 0.2


def classical_train_transforms(size: int = 224):
    """Shared classical baseline: rotation + brightness, ImageNet norm."""
    from torchvision import transforms

    from biosentinel.training_recipe import IMAGENET_MEAN, IMAGENET_STD

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomRotation(ROTATION_DEGREES),
            transforms.ColorJitter(brightness=BRIGHTNESS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def cap_synthetic_count(requested: int, cap: int = SYNTHETIC_CAP_DETECTION) -> int:
    """Cap synthetic images for detection experiments."""
    return max(0, min(int(requested), int(cap)))


def keep_synthetic(heldout_gain: float) -> bool:
    """Keep synthetic data only on strict real held-out gain."""
    return float(heldout_gain) > 0.0
