# Fichier : packaging/app/benchmark.py
"""Étape 5 — latence d'inférence : pipeline scikit-learn vs ONNX Runtime.

Usage (racine du repo) : uv run python packaging/app/benchmark.py
Écrit aussi le tableau dans reports/benchmark.md.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

N = 1000

cfg = load_config(ROOT / "dataset.yaml")
meta = json.loads((ROOT / "packaging/model/metadata.json").read_text())
X, _ = cfg.read_xy(ROOT / "data/processed/test.csv")  # numériques en float64


def onnx_feed(df: pd.DataFrame) -> dict:
    feed = {c: df[[c]].astype(np.float64).to_numpy() for c in cfg.numeric}
    feed |= {c: df[[c]].fillna("").astype(str).to_numpy(dtype=object) for c in cfg.categorical}
    return feed


def bench(name, fn):
    for _ in range(10):
        fn()  # warmup
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {
        "name": name,
        "p50": np.percentile(a, 50),
        "p95": np.percentile(a, 95),
        "p99": np.percentile(a, 99),
    }


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
sk = mlflow.sklearn.load_model(f"models:/{meta['model_name']}/{meta['model_version']}")
ox = ort.InferenceSession(
    str(ROOT / "packaging/model/model.onnx"), providers=["CPUExecutionProvider"]
)

x1, x100 = X.head(1), X.head(100)
f1, f100 = onnx_feed(x1), onnx_feed(x100)
results = [
    bench("sklearn_pipeline", lambda: sk.predict_proba(x1)),
    bench("onnx_python", lambda: ox.run(None, f1)),
    bench(f"onnx_batch_{len(x100)}", lambda: ox.run(None, f100)),
]

lines = [f"{'Backend':<20} {'P50':>8} {'P95':>8} {'P99':>8} (ms)"]
lines += [f"{r['name']:<20} {r['p50']:>8.3f} {r['p95']:>8.3f} {r['p99']:>8.3f}" for r in results]
print("\n".join(lines))
speedup = results[0]["p50"] / results[1]["p50"]
print(f"ONNX vs sklearn (P50, 1 row): x{speedup:.1f}")

report = ROOT / "reports" / "benchmark.md"
report.parent.mkdir(exist_ok=True)
rows = "\n".join(
    f"| {r['name']} | {r['p50']:.3f} | {r['p95']:.3f} | {r['p99']:.3f} |" for r in results
)
report.write_text(
    f"# Benchmark de latence — {cfg.name}, modèle v{meta['model_version']}\n\n"
    f"{N} appels par backend, après 10 appels de chauffe · "
    f"{platform.processor() or platform.machine()} · Python {platform.python_version()}\n\n"
    "| Backend | P50 (ms) | P95 (ms) | P99 (ms) |\n|---|---|---|---|\n"
    f"{rows}\n\nAccélération ONNX vs scikit-learn (P50, 1 ligne) : x{speedup:.1f}\n"
)
print(f"Report written: {report.relative_to(ROOT)}")
