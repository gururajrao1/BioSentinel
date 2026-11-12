"""Smoke tests for the runnable training entry points."""

import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from biosentinel.trainer import build_scheduler, train_classifier
from compare import format_markdown_table, run_comparison


def _make_dataset(root: Path, classes=("burn", "necrosis"), n_train=4, n_val=2):
    for split, n in (("train", n_train), ("val", n_val), ("test", n_val)):
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.new("RGB", (48, 48), color=(i * 40, 100, 30)).save(d / f"{i}.jpg")


def test_train_classifier_smoke(tmp_path):
    data = tmp_path / "dataset"
    _make_dataset(data)
    out = tmp_path / "outputs"
    metrics = train_classifier(
        str(data),
        arch="resnet18",
        img_size=32,
        batch_size=2,
        epochs_stage1=1,
        epochs_stage2=1,
        pretrained=False,
        output_dir=str(out),
    )
    assert set(metrics["val"]) >= {
        "accuracy",
        "per_class_recall",
        "per_class_precision",
        "per_class_f1",
        "macro_f1",
        "mcc",
        "auroc",
        "confusion_matrix",
        "top_confusions",
    }
    assert set(metrics["calibration"]) >= {
        "temperature",
        "ece_before",
        "ece_after",
    }
    assert (out / "best_resnet18.pth").exists()
    assert json.loads((out / "metrics_resnet18.json").read_text())["arch"] == "resnet18"


def test_compare_writes_csv_and_markdown(tmp_path):
    data = tmp_path / "dataset"
    _make_dataset(data)
    rows = run_comparison(
        str(data),
        ["mobilenet_v3_small"],
        output_dir=str(tmp_path / "compare"),
        img_size=32,
        batch_size=2,
        epochs_stage1=1,
        epochs_stage2=1,
        pretrained=False,
    )
    assert len(rows) == 1 and rows[0]["arch"] == "mobilenet_v3_small"
    with open(tmp_path / "compare" / "results.csv") as f:
        assert [r["arch"] for r in csv.DictReader(f)] == ["mobilenet_v3_small"]
    md = format_markdown_table(rows)
    assert "mobilenet_v3_small" in md and md.startswith("| arch |")


def test_compare_appends_soft_vote_ensemble(tmp_path):
    from compare import run_comparison

    data = tmp_path / "dataset"
    _make_dataset(data)
    rows = run_comparison(
        str(data),
        ["resnet18", "mobilenet_v3_small"],
        output_dir=str(tmp_path / "compare2"),
        img_size=32,
        batch_size=2,
        epochs_stage1=1,
        epochs_stage2=1,
        pretrained=False,
    )
    assert [r["arch"] for r in rows] == [
        "resnet18",
        "mobilenet_v3_small",
        "ensemble(n=2)",
    ]
    assert rows[-1]["val_macro_f1"] >= 0.0


def test_amp_and_early_stopping_paths(tmp_path):
    import json

    data = tmp_path / "dataset"
    _make_dataset(data)
    out = tmp_path / "outputs-amp"
    metrics = train_classifier(
        str(data),
        arch="resnet18",
        img_size=32,
        batch_size=2,
        epochs_stage1=2,
        epochs_stage2=2,
        pretrained=False,
        use_amp=True,
        patience=1,
        output_dir=str(out),
    )
    run = json.loads((out / "run.json").read_text())
    assert run["params"]["epochs_run"] <= 4
    assert run["params"]["use_amp"] is True
    assert (out / "best_resnet18.pth").exists()
    assert metrics["val"]["macro_f1"] >= 0.0


def test_focal_loss_trains(tmp_path):
    data = tmp_path / "dataset"
    _make_dataset(data)
    out = tmp_path / "outputs-focal"
    metrics = train_classifier(
        str(data),
        arch="resnet18",
        img_size=32,
        batch_size=2,
        epochs_stage1=1,
        epochs_stage2=1,
        pretrained=False,
        loss_name="focal",
        patience=None,
        output_dir=str(out),
    )
    assert metrics["val"]["accuracy"] >= 0.0
    assert (out / "best_resnet18.pth").exists()


def test_scheduler_warms_up_then_decays():
    import torch

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sched = build_scheduler(opt, warmup_steps=2, total_steps=10)
    lrs = []
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[1] > lrs[0]  # warming up
    assert lrs[-1] < lrs[2]  # cosine decay after warmup
