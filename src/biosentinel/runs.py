"""Per-run manifest: params + metrics + data fingerprint (§4 MLOps).

One JSON file per training run so any checkpoint can be traced back to the
code inputs and data it came from. Keep human-in-the-loop gates on top.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os


def data_fingerprint(data_dir: str) -> str:
    """Stable sha256 over sorted relative paths and file sizes."""
    entries = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            entries.append(
                f"{os.path.relpath(path, data_dir)}:{os.path.getsize(path)}"
            )
    entries.sort()
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def _sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def log_run(output_dir: str, params: dict, metrics: dict, data_dir: str) -> str:
    """Write run.json and return its path."""
    manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "params": _sanitize(params),
        "metrics": _sanitize(metrics),
        "data_dir": os.path.abspath(data_dir),
        "data_fingerprint": data_fingerprint(data_dir),
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
