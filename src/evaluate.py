# Fichier : src/evaluate.py
"""Évalue le modèle local (models/model) sur le jeu de test -> reports/metrics.json.

Stage DVC « evaluate » : ne dépend que de fichiers versionnés (pas du serveur MLflow),
pour que `dvc metrics show` / `dvc metrics diff` comparent les versions du pipeline.
Le seuil de décision est celui choisi par src/train.py (métadonnées du modèle).
"""

import json
from pathlib import Path

import mlflow
from mlflow.models import Model
from sklearn.metrics import confusion_matrix

from src.dataset_config import load_config
from src.train import MODEL_DIR, TEST_CSV, compute_metrics

METRICS_JSON = Path("reports/metrics.json")


def main() -> None:
    cfg = load_config()
    X_test, y_test = cfg.read_xy(TEST_CSV)
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    threshold = Model.load(str(MODEL_DIR)).metadata["decision_threshold"]

    proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(cfg, y_test, proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= threshold).astype(int)).ravel()
    report = {
        **{k: round(float(v), 4) for k, v in metrics.items()},
        "decision_threshold": round(threshold, 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_test)),
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{cfg.primary_metric} = {report[cfg.primary_metric]} -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
