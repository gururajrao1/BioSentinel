"""Head-to-head comparison across architectures (§3, using §1 + §4).

Trains each arch with the same two-stage recipe and data splits, then writes
results.csv + results.md so the pick is made on AUROC + per-class recall,
never accuracy alone.

Example:
    python scripts/compare.py --data dataset \\
        --arches resnet18 efficientnet_b4 mobilenet_v3_small
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.models import EDGE_ARCHES
from biosentinel.trainer import train_classifier


def format_markdown_table(rows: list[dict]) -> str:
    header = "| arch | val_acc | val_auroc | val_macro_f1 | val_mcc | val_recall | test_acc |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in rows:
        recall = ", ".join(f"{v:.2f}" for v in r["val_recall"])
        lines.append(
            f"| {r['arch']} | {r['val_acc']:.3f} | {r['val_auroc']:.3f}"
            f" | {r['val_macro_f1']:.3f} | {r['val_mcc']:.3f}"
            f" | {recall} | {r['test_acc']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def ensemble_row(
    data_dir: str,
    arches: list[str],
    output_dir: str,
    classes: list[str],
    img_size: int = 224,
    batch_size: int = 32,
) -> dict:
    """Soft-vote the trained arch checkpoints on the val split."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from biosentinel import evaluation as ev
    from biosentinel.models import build_classifier, soft_vote
    from biosentinel.training_recipe import make_eval_transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        datasets.ImageFolder(
            os.path.join(data_dir, "val"),
            transform=make_eval_transforms(img_size),
        ),
        batch_size=batch_size,
    )
    all_probs, labels = [], None
    with torch.no_grad():
        for arch in arches:
            model = build_classifier(arch, num_classes=len(classes), pretrained=False)
            model.load_state_dict(
                torch.load(
                    os.path.join(output_dir, arch, f"best_{arch}.pth"),
                    map_location=device,
                )
            )
            model = model.to(device).eval()
            probs, batch_labels = [], []
            for images, targets in loader:
                probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
                batch_labels.extend(targets.tolist())
            all_probs.append(torch.cat(probs))
            labels = batch_labels
    from sklearn.metrics import confusion_matrix

    mean = torch.stack(all_probs).mean(dim=0)
    voted = soft_vote([p.tolist() for p in all_probs])
    summary = ev.extended_metrics(labels, voted)
    return {
        "arch": f"ensemble(n={len(arches)})",
        "val_acc": float(np.mean([v == t for v, t in zip(voted, labels)])),
        "val_auroc": ev.multiclass_auroc(labels, mean.tolist()),
        "val_macro_f1": summary["macro_f1"],
        "val_mcc": summary["mcc"],
        "val_recall": summary["per_class_recall"],
        "test_acc": float("nan"),
    }


def run_comparison(
    data_dir: str,
    arches: list[str],
    output_dir: str = "outputs/compare",
    **train_kwargs,
) -> list[dict]:
    rows = []
    classes: list[str] | None = None
    for arch in arches:
        arch_dir = os.path.join(output_dir, arch)
        metrics = train_classifier(
            data_dir=data_dir, arch=arch, output_dir=arch_dir, **train_kwargs
        )
        classes = metrics["classes"]
        rows.append(
            {
                "arch": arch,
                "val_acc": metrics["val"]["accuracy"],
                "val_auroc": metrics["val"]["auroc"],
                "val_macro_f1": metrics["val"]["macro_f1"],
                "val_mcc": metrics["val"]["mcc"],
                "val_recall": metrics["val"]["per_class_recall"],
                "test_acc": metrics.get("test", {}).get("accuracy", float("nan")),
            }
        )
    if len(arches) > 1 and classes is not None:
        rows.append(
            ensemble_row(
                data_dir,
                list(arches),
                output_dir,
                classes,
                img_size=train_kwargs.get("img_size", 224),
                batch_size=train_kwargs.get("batch_size", 32),
            )
        )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "val_acc",
                "val_auroc",
                "val_macro_f1",
                "val_mcc",
                "val_recall",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, "results.md"), "w") as f:
        f.write("# Architecture comparison\n\n")
        f.write(format_markdown_table(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare architectures head-to-head.")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--arches", nargs="+", default=list(EDGE_ARCHES))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=2)
    parser.add_argument("--epochs-stage2", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--loss", default="ce", choices=["ce", "focal"])
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--output-dir", default="outputs/compare")
    args = parser.parse_args()
    for arch in args.arches:
        if arch not in EDGE_ARCHES:
            parser.error(f"unknown arch {arch!r}; choose from {list(EDGE_ARCHES)}")

    rows = run_comparison(
        data_dir=args.data,
        arches=list(args.arches),
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        pretrained=not args.no_pretrained,
        use_amp=not args.no_amp,
        patience=args.patience,
        loss_name=args.loss,
        balanced_sampler=args.balanced_sampler,
    )
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()
