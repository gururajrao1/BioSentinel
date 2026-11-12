"""Tests for the per-run manifest."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.runs import data_fingerprint, log_run


def test_fingerprint_stable_and_sensitive(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"123")
    first = data_fingerprint(str(tmp_path))
    assert data_fingerprint(str(tmp_path)) == first
    (tmp_path / "b.jpg").write_bytes(b"4567")
    assert data_fingerprint(str(tmp_path)) != first


def test_log_run_writes_manifest(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.jpg").write_bytes(b"1")
    path = log_run(
        str(tmp_path / "out"),
        params={"arch": "resnet18"},
        metrics={"val_acc": 0.8, "auroc": float("nan")},
        data_dir=str(tmp_path / "data"),
    )
    manifest = json.loads(Path(path).read_text())
    assert manifest["params"] == {"arch": "resnet18"}
    assert manifest["metrics"]["val_acc"] == 0.8
    assert manifest["metrics"]["auroc"] is None
    assert len(manifest["data_fingerprint"]) == 64
