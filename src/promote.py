# Fichier : src/promote.py
"""Enregistre un run dans le Model Registry et le promeut avec des alias MLflow.

  1. choisit le meilleur run de l'expérience « scoring » selon metrics.primary
     (dataset.yaml) — ou le run donné par --run-id, ou le meilleur parmi --filter ;
  2. l'enregistre comme nouvelle version de « scoring-model » (sauf s'il l'est déjà) ;
  3. pose l'alias @challenger sur cette version ;
  4. pose l'alias @champion SEULEMENT si la métrique atteint metrics.promotion_threshold.
Les alias remplacent les « stages » (Staging/Production) dépréciés depuis MLflow 2.9.
"""

import argparse
import os

import mlflow
from mlflow import MlflowClient

from src.dataset_config import load_config
from src.train import EXPERIMENT

MODEL_NAME = "scoring-model"
# Métriques loggées par src/train.py pour lesquelles « plus grand = meilleur »
HIGHER_IS_BETTER = {"roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="run à enregistrer (défaut : le meilleur run)")
    parser.add_argument(
        "--filter",
        default="",
        help="filtre MLflow sur les runs candidats, ex. \"tags.launched_by = 'dvc'\"",
    )
    return parser.parse_args()


def find_run(client: MlflowClient, metric: str, run_id: str | None, extra_filter: str):
    """Le run demandé, sinon le meilleur run terminé (à égalité : le plus récent)."""
    if run_id:
        return client.get_run(run_id)
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"Expérience « {EXPERIMENT} » introuvable : lancez d'abord src.train")
    filters = ["attributes.status = 'FINISHED'"] + ([extra_filter] if extra_filter else [])
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=" and ".join(filters),
        order_by=[f"metrics.{metric} DESC", "attributes.start_time DESC"],
        max_results=1,
    )
    if not runs or metric not in runs[0].data.metrics:
        raise SystemExit(f"Aucun run avec la métrique « {metric} » ({' and '.join(filters)})")
    return runs[0]


def main() -> None:
    args = parse_args()
    cfg = load_config()
    metric, threshold = cfg.primary_metric, cfg.promotion_threshold
    if metric not in HIGHER_IS_BETTER:
        raise SystemExit(f"metrics.primary doit être l'une de {sorted(HIGHER_IS_BETTER)}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    run = find_run(client, metric, args.run_id, args.filter)
    score = run.data.metrics[metric]

    # Une seule version par run : relancer le script ne crée pas de doublon
    existing = client.search_model_versions(
        f"name = '{MODEL_NAME}' and run_id = '{run.info.run_id}'"
    )
    if existing:
        version = existing[0].version
        print(f"Run {run.info.run_name} déjà enregistré : version {version}")
    else:
        # MLflow 3 : le modèle loggé par le run est un « LoggedModel » (models:/m-...)
        model_id = run.outputs.model_outputs[0].model_id
        mv = mlflow.register_model(f"models:/{model_id}", MODEL_NAME)
        version = mv.version
        print(f"Run {run.info.run_name} enregistré : {MODEL_NAME} version {version}")

    passed = score >= threshold
    tags = {
        metric: f"{score:.4f}",
        "validation_status": "passed" if passed else "failed",
        "data_md5": run.data.tags.get("data_md5", "unknown"),
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)

    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"  @challenger -> version {version} ({metric} = {score:.4f})")
    if passed:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        print(f"  @champion   -> version {version} ({metric} >= seuil {threshold})")
    else:
        try:
            current = client.get_model_version_by_alias(MODEL_NAME, "champion").version
        except mlflow.exceptions.MlflowException:
            current = "aucun"
        print(f"  champion inchangé (version {current}) : {metric} < seuil {threshold}")


if __name__ == "__main__":
    main()
