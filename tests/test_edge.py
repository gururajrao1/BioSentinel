"""Tests for §5 edge export and quantization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from biosentinel.edge import (
    calibration_subset,
    export_onnx,
    measure_latency_ms,
    quantize_dynamic,
    serving_transforms,
    state_size_kb,
    trace_torchscript,
)
from biosentinel.models import build_classifier


def test_torchscript_trace_roundtrip():
    model = build_classifier("mobilenet_v3_small", num_classes=4)
    payload = trace_torchscript(model, img_size=64)
    assert len(payload) > 1000


def test_dynamic_quantize_keeps_forward_shape():
    import torch

    model = quantize_dynamic(build_classifier("resnet18", num_classes=4))
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 4)


def test_serving_matches_eval_transforms():
    from PIL import Image

    from biosentinel.training_recipe import make_eval_transforms

    img = Image.new("RGB", (64, 64), color=(10, 200, 90))
    a = serving_transforms(64)(img)
    b = make_eval_transforms(64)(img)
    assert torch.equal(a, b)


def test_calibration_subset_is_deterministic():
    ds = list(range(10))
    assert [i for i in calibration_subset(ds, n=4, seed=1).indices] == [
        i for i in calibration_subset(ds, n=4, seed=1).indices
    ]


def test_onnx_export_when_available():
    pytest.importorskip("onnx")
    payload = export_onnx(build_classifier("mobilenet_v3_small"), img_size=32)
    assert len(payload) > 1000


def test_size_and_latency_are_positive():
    model = build_classifier("mobilenet_v3_small", num_classes=4)
    assert state_size_kb(model) > 0
    assert measure_latency_ms(model, img_size=64, repeats=3) > 0
