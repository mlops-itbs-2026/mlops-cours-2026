# Fichier : packaging/app/verify_onnx.py
"""Étape 1.3 — vérifie que le modèle ONNX prédit comme le modèle scikit-learn.

Usage (racine du repo) : uv run python packaging/app/verify_onnx.py
"""

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

TOL = 1e-4  # écart absolu toléré sur les probabilités (float32 vs float64)

cfg = load_config(ROOT / "dataset.yaml")
meta = json.loads((ROOT / "packaging/model/metadata.json").read_text())
X, _ = cfg.read_xy(ROOT / "data/processed/test.csv")  # numériques en float64

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
# La version exacte qui a été exportée (et non l'alias, qui a pu bouger depuis)
skm = mlflow.sklearn.load_model(f"models:/{meta['model_name']}/{meta['model_version']}")
sess = ort.InferenceSession(str(ROOT / "packaging/model/model.onnx"))

feed = {c: X[[c]].astype(np.float64).to_numpy() for c in cfg.numeric}
feed |= {c: X[[c]].fillna("").astype(str).to_numpy(dtype=object) for c in cfg.categorical}

p1 = skm.predict_proba(X)[:, 1]
p2 = sess.run(["probabilities"], feed)[0][:, 1]
t = meta["threshold"]
max_diff = float(np.abs(p1 - p2).max())
same = float(((p1 >= t) == (p2 >= t)).mean())
print(
    f"{len(X)} rows · max |p_sklearn - p_onnx| = {max_diff:.2e} · "
    f"same decision (threshold {t:.3f}) : {same:.1%}"
)
assert max_diff < TOL, "Divergence detected!"
print("OK: predictions match.")
