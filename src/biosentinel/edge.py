"""Edge deployment (§5 of research-backed plan).

Ship a TorchScript/ONNX-INT8 build for offline field use: calibrate PTQ on
representative wound images and prefer QAT for the clinical-grade model.
Helpers stay framework-light: trace → dynamic-quantize → size/latency check.
"""

from __future__ import annotations

import io
import os
import time

import torch
import torch.nn as nn


def trace_torchscript(model: nn.Module, img_size: int = 128) -> bytes:
    model = model.cpu().eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.randn(1, 3, img_size, img_size))
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


def quantize_dynamic(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(
        model.cpu().eval(), {nn.Linear}, dtype=torch.qint8
    )


def serving_transforms(img_size: int = 128):
    """Serving transforms. Must stay identical to training eval transforms."""
    from biosentinel.training_recipe import make_eval_transforms

    return make_eval_transforms(img_size)


def export_onnx(model: nn.Module, img_size: int = 128) -> bytes:
    """Export to ONNX for runtimes with INT8 kernels (e.g. ORT + XNNPACK)."""
    try:
        import torch.onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export needs the onnx package: pip install onnx"
        ) from exc
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy,
        buf,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return buf.getvalue()


def calibration_subset(dataset, n: int = 200, seed: int = 42):
    """Representative deployment-environment sample for PTQ calibration."""
    import random

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
    return torch.utils.data.Subset(dataset, sorted(indices))


def state_size_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024


def measure_latency_ms(model: nn.Module, img_size: int = 128, repeats: int = 20) -> float:
    model = model.cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        end = time.perf_counter()
    return (end - start) / repeats * 1000


def write_artifact(path: str, payload: bytes) -> float:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    return os.path.getsize(path) / 1024
