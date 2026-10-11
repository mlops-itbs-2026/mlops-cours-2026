# Fichier : packaging/app/export_onnx.py
"""Étape 1 — récupère scoring-model@champion dans MLflow et l'exporte en ONNX.

Usage (depuis la racine du repo, après `set -a && source .env && set +a`) :
    uv run python packaging/app/export_onnx.py
Produit packaging/model/model.onnx et packaging/model/metadata.json.
"""

import copy
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient
from skl2onnx import to_onnx
from skl2onnx.common.data_types import DoubleTensorType, StringTensorType
from skl2onnx.sklapi import CastTransformer
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[2]  # racine du repo
sys.path.append(str(ROOT))  # rend « src » importable quel que soit le dossier courant
from src.dataset_config import load_config  # noqa: E402

MODEL_NAME = os.getenv("MODEL_NAME", "scoring-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_DIR = ROOT / "packaging" / "model"
TARGET_OPSET = 17

cfg = load_config(ROOT / "dataset.yaml")

# Seule variable nécessaire : le serveur MLflow fait proxy des artefacts S3
# (aucun identifiant S3 côté client).
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")  # MLflow 3 : skops
print(
    f"Loaded models:/{MODEL_NAME}@{MODEL_ALIAS} -> version {mv.version} "
    f"({type(model).__name__}: {' -> '.join(type(s).__name__ for _, s in model.steps)})"
)

# 1. Types d'entrée dérivés de dataset.yaml : un tenseur [N, 1] par colonne, nommé comme
#    la colonne (le ColumnTransformer sélectionne les colonnes par leur nom).
#    float64 pour les numériques : voir onnx_ready(), point (b).
initial_types = [(c, DoubleTensorType([None, 1])) for c in cfg.numeric] + [
    (c, StringTensorType([None, 1])) for c in cfg.categorical
]


# 2. Copie du pipeline réécrite pour une conversion fidèle (le modèle MLflow n'est pas modifié).
def onnx_ready(pipeline):
    pipeline = copy.deepcopy(pipeline)
    pre = next(step for _, step in pipeline.steps if isinstance(step, ColumnTransformer))
    if cfg.categorical:
        # (a) skl2onnx ne convertit pas un imputer de chaînes dont la valeur manquante est
        #     NaN : dans le graphe ONNX, « manquant » sera la chaîne vide "".
        pre.named_transformers_["cat"].named_steps["impute"].missing_values = ""
    if cfg.numeric:
        # (b) Les arbres de scikit-learn comparent des float32 à des seuils situés SUR des
        #     valeurs observées : standardiser en float32 (au lieu de float64 puis float32)
        #     décale d'un arrondi et change de branche. On standardise donc en float64,
        #     on convertit en float32, puis on impute avec la médiane standardisée.
        num = pre.named_transformers_["num"]
        imputer, scaler = num.named_steps["impute"], num.named_steps["scale"]
        imputer.statistics_ = (imputer.statistics_ - scaler.mean_) / scaler.scale_
        to_float32 = CastTransformer(dtype=np.float32).fit(np.zeros((1, len(cfg.numeric))))
        num.steps = [("scale", scaler), ("float32", to_float32), ("impute", imputer)]
    return pipeline


onnx_model = to_onnx(
    onnx_ready(model),
    initial_types=initial_types,
    target_opset=TARGET_OPSET,
    options={"zipmap": False},  # probabilités en tenseur [N, 2] plutôt qu'en liste de dicts
)


# 3. Seuil de décision : celui calculé et loggé par l'entraînement (Lab 1, param MLflow
#    `decision_threshold`). Repli pour un run plus ancien : le même calcul, cfg.cost_threshold(k).
def decision_threshold() -> tuple[float, str]:
    logged = client.get_run(mv.run_id).data.params.get("decision_threshold")
    if logged is not None:
        return float(logged), "param decision_threshold du run"
    k = 1.0
    if getattr(model.steps[-1][1], "class_weight", None) == "balanced":
        _, y = cfg.read_xy(ROOT / "data/processed/train.csv")
        k = (y == 0).sum() / (y == 1).sum()  # poids des positifs / poids des négatifs
    return cfg.cost_threshold(float(k)), "cfg.cost_threshold (repli)"


threshold, threshold_source = decision_threshold()

MODEL_DIR.mkdir(parents=True, exist_ok=True)
onnx_path = MODEL_DIR / "model.onnx"
onnx_bytes = onnx_model.SerializeToString()
onnx_path.write_bytes(onnx_bytes)

# 4. Métadonnées lues par l'API (/metadata) et reprises en LABEL de l'image.
metadata = {
    "dataset": cfg.name,
    "model_name": MODEL_NAME,
    "model_alias": MODEL_ALIAS,
    "model_version": str(mv.version),
    "run_id": mv.run_id,
    "target": cfg.target,
    "positive_class": str(cfg.positive_class),
    "threshold": round(threshold, 6),
    "threshold_source": threshold_source,
    "features": {"numeric": cfg.numeric, "categorical": cfg.categorical},
    "onnx_opset": TARGET_OPSET,
    "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
    "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

size_kb = onnx_path.stat().st_size / 1024
print(f"ONNX exported: {size_kb:.1f} kB -> {onnx_path.relative_to(ROOT)}")
print(
    f"Inputs: {len(cfg.numeric)} numeric + {len(cfg.categorical)} categorical "
    f"· threshold={threshold:.3f} ({threshold_source})"
)
