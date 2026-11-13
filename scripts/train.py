"""Train one wound classifier with the §1 two-stage recipe.

Example:
    python scripts/train.py --data dataset --arch resnet18
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.models import EDGE_ARCHES
from biosentinel.trainer import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BioSentinel classifier.")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--arch", default="resnet18", choices=list(EDGE_ARCHES))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=3)
    parser.add_argument("--epochs-stage2", type=int, default=3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-late", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--loss", default="ce", choices=["ce", "focal"])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    metrics = train_classifier(
        data_dir=args.data,
        arch=args.arch,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr_head=args.lr_head,
        lr_late=args.lr_late,
        optimizer_name=args.optimizer,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        calibrate=not args.no_calibrate,
        pretrained=not args.no_pretrained,
        use_amp=not args.no_amp,
        patience=args.patience,
        balanced_sampler=args.balanced_sampler,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    val = metrics["val"]
    print(f"arch={args.arch} val_acc={val['accuracy']:.3f} auroc={val['auroc']:.3f}")
    print(f"per-class recall: {val['per_class_recall']}")
    if "test" in metrics:
        print(f"test_acc={metrics['test']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
