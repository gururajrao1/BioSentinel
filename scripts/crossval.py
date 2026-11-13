"""K-fold stability check: same recipe, different splits.

Answers "was it a lucky split?" with mean±std of val macro-F1 across folds.
Uses train+val pooled and stratified; the locked test set stays untouched.

Example:
    python scripts/crossval.py --data data/dataset-big --arch convnext_tiny \\
        --folds 3 --loss focal
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.trainer import train_classifier


def collect_samples(data_dir: str) -> tuple[list[str], list[str], list[str]]:
    """(paths, classes, class_names) pooled from train+val splits."""
    paths, labels, names = [], [], []
    for split in ("train", "val"):
        base = os.path.join(data_dir, split)
        if not os.path.isdir(base):
            continue
        for cls in sorted(os.listdir(base)):
            cls_dir = os.path.join(base, cls)
            if not os.path.isdir(cls_dir):
                continue
            if cls not in names:
                names.append(cls)
            for f in sorted(os.listdir(cls_dir)):
                paths.append(os.path.join(cls_dir, f))
    names.sort()
    labels = [names.index(os.path.basename(os.path.dirname(p))) for p in paths]
    return paths, labels, names


def make_fold_layout(
    paths: list[str],
    labels: list[int],
    class_names: list[str],
    n_splits: int,
    seed: int,
    out_dir: str,
) -> list[str]:
    """Copy files into fold{i}/{train,val}/{class}/ dirs. Returns fold dirs."""
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_dirs = []
    y = np.asarray(labels)
    for i, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_dir = os.path.join(out_dir, f"fold{i}")
        for idx, split in ((tr, "train"), (te, "val")):
            for j in idx:
                dest = os.path.join(fold_dir, split, class_names[labels[j]])
                os.makedirs(dest, exist_ok=True)
                shutil.copy(paths[j], os.path.join(dest, os.path.basename(paths[j])))
        fold_dirs.append(fold_dir)
    return fold_dirs


def run_crossval(
    data_dir: str,
    arch: str,
    n_splits: int = 3,
    seed: int = 42,
    output_dir: str = "outputs/crossval",
    **train_kwargs,
) -> list[dict]:
    """Train one fold per split; return per-fold val metric rows."""
    paths, labels, names = collect_samples(data_dir)
    fold_dirs = make_fold_layout(paths, labels, names, n_splits, seed, output_dir)
    rows = []
    for i, fold_dir in enumerate(fold_dirs):
        metrics = train_classifier(
            data_dir=fold_dir,
            arch=arch,
            output_dir=os.path.join(output_dir, arch, f"fold{i}"),
            seed=seed + i,
            **train_kwargs,
        )
        rows.append(
            {
                "fold": i,
                "val_acc": metrics["val"]["accuracy"],
                "val_auroc": metrics["val"]["auroc"],
                "val_macro_f1": metrics["val"]["macro_f1"],
            }
        )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"crossval_{arch}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "val_acc", "val_auroc", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold stability check.")
    parser.add_argument("--data", default="data/dataset-big")
    parser.add_argument("--arch", default="convnext_tiny")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-stage1", type=int, default=5)
    parser.add_argument("--epochs-stage2", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loss", default="focal", choices=["ce", "focal"])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/crossval")
    args = parser.parse_args()

    rows = run_crossval(
        args.data,
        args.arch,
        n_splits=args.folds,
        seed=args.seed,
        output_dir=args.output_dir,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        loss_name=args.loss,
        patience=args.patience,
    )
    f1s = [r["val_macro_f1"] for r in rows]
    print(f"macro-F1: mean={np.mean(f1s):.3f} std={np.std(f1s):.3f} n={len(f1s)}")


if __name__ == "__main__":
    main()
