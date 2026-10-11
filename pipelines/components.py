# Fichier : pipelines/components.py
"""Composants KFP du pipeline de scoring (Lab 2).

Chaque composant tourne dans l'image « projet » PIPELINE_IMAGE (pipelines/Dockerfile), qui
contient le code du Lab 1. Un composant ne recopie AUCUNE logique métier : il relie ses
artefacts typés (Dataset, Model, Metrics) aux fichiers d'une stage de dvc.yaml, puis lance
la même commande que DVC (python -m src.<étape>) grâce à pipelines/stage.py.

Rappel KFP : seul le CODE de chaque fonction est envoyé au pod. Les imports se font donc
DANS la fonction (ils sont résolus dans l'image) et aucune valeur par défaut d'un
paramètre ne doit dépendre d'une constante de ce module.
"""

import os

from kfp import dsl
from kfp.dsl import Artifact, ClassificationMetrics, Dataset, Input, Metrics, Model, Output

# Image construite et poussée à l'étape 2.4. Nouveau code dans l'image = nouveau tag.
# Surcharge possible à la compilation : PIPELINE_IMAGE=… python -m pipelines.scoring_pipeline
PIPELINE_IMAGE = os.getenv("PIPELINE_IMAGE", "localhost:5001/scoring-pipeline:1.0.0")
# Serveur MLflow du cours vu depuis un pod (DNS interne <service>.<namespace>)
MLFLOW_URI = "http://mlflow.mlflow:5000"


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def ingest_data(raw_data: Output[Dataset]) -> None:
    """Stage « download » : python -m src.download produit le fichier brut."""
    import pandas as pd

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import file_md5

    cfg = load_config()
    run_stage("src.download", outputs={cfg.raw_path: raw_data.path})
    raw_data.metadata["dataset"] = cfg.name
    raw_data.metadata["rows"] = len(pd.read_csv(raw_data.path))
    raw_data.metadata["md5"] = file_md5(raw_data.path)  # le même hash que dans dvc.lock


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def validate_data(
    raw_data: Input[Dataset],
    report: Output[Artifact],
    checks: Output[Metrics],
) -> None:
    """Gate : python -m src.validate_data ; un contrôle en échec fait échouer le step."""
    import json
    from pathlib import Path

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.validate_data import REPORT_JSON

    cfg = load_config()
    run_stage(
        "src.validate_data",
        inputs={cfg.raw_path: raw_data.path},
        outputs={str(REPORT_JSON): report.path},  # reports/validation.json
    )
    summary = json.loads(Path(report.path).read_text())
    for name in ("n_checks", "n_failed", "rows", "positive_rate"):
        checks.log_metric(name, summary[name])


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def prepare_features(
    raw_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
) -> None:
    """Stage « prepare » : python -m src.prepare (split stratifié décrit par dataset.yaml)."""
    import pandas as pd

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import TEST_CSV, TRAIN_CSV

    cfg = load_config()
    run_stage(
        "src.prepare",
        inputs={cfg.raw_path: raw_data.path},
        outputs={str(TRAIN_CSV): train_data.path, str(TEST_CSV): test_data.path},
    )
    for split in (train_data, test_data):
        y = cfg.encode_target(pd.read_csv(split.path)[cfg.target])
        split.metadata["rows"] = len(y)
        split.metadata["positive_rate"] = round(float(y.mean()), 4)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def train_model(
    raw_data: Input[Dataset],
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Output[Model],
    model_type: str,
    C: float,
    class_weight: str,
    n_estimators: int,
    max_depth: str,
    learning_rate: float,
    random_state: int,
    mlflow_uri: str,
    kfp_run_id: str,
) -> str:
    """Stage « train » : python -m src.train ; renvoie l'identifiant du run MLflow."""
    from mlflow import MlflowClient
    from mlflow.models import Model as MlflowModel

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import MODEL_DIR, TEST_CSV, TRAIN_CSV

    cfg = load_config()
    run_stage(
        "src.train",
        args=["--model", model_type, "--C", C, "--class_weight", class_weight,
              "--n_estimators", n_estimators, "--max_depth", max_depth,
              "--learning_rate", learning_rate, "--random_state", random_state],
        # le fichier brut sert au tag data_md5 (le même hash que dans dvc.lock)
        inputs={cfg.raw_path: raw_data.path, str(TRAIN_CSV): train_data.path,
                str(TEST_CSV): test_data.path},
        outputs={str(MODEL_DIR): model.path},
        env={"MLFLOW_TRACKING_URI": mlflow_uri, "LAUNCHED_BY": "kfp"},
    )  # fmt: skip
    metadata = MlflowModel.load(model.path).metadata  # écrites par src/train.py
    run_id = metadata["mlflow_run_id"]
    # Lineage : depuis le run MLflow, on retrouve le run KFP qui l'a produit
    MlflowClient(mlflow_uri).set_tag(run_id, "kfp_run_id", kfp_run_id)
    model.metadata["mlflow_run_id"] = run_id
    model.metadata["model_type"] = model_type
    model.metadata["decision_threshold"] = round(metadata["decision_threshold"], 4)
    return run_id


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    report: Output[Artifact],
    metrics: Output[Metrics],
    confusion: Output[ClassificationMetrics],
    run_id: str,
    mlflow_uri: str,
) -> float:
    """Stage « evaluate » : python -m src.evaluate ; métriques dans l'UI KFP et MLflow."""
    import json
    from pathlib import Path

    import pandas as pd
    from mlflow import MlflowClient

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.evaluate import METRICS_JSON
    from src.train import MODEL_DIR, TEST_CSV

    cfg = load_config()
    run_stage(
        "src.evaluate",
        inputs={str(MODEL_DIR): model.path, str(TEST_CSV): test_data.path},
        outputs={str(METRICS_JSON): report.path},  # reports/metrics.json
    )
    results = json.loads(Path(report.path).read_text())
    for name, value in results.items():
        if isinstance(value, int | float):
            metrics.log_metric(name, value)

    # Matrice de confusion au seuil de décision, avec les libellés réels de la cible
    positive = str(cfg.positive_class)
    others = sorted(set(pd.read_csv(test_data.path)[cfg.target].astype(str)) - {positive})
    negative = others[0] if len(others) == 1 else f"non {positive}"
    c = results["confusion"]
    confusion.log_confusion_matrix([negative, positive], [[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    # Le même rapport que reports/metrics.json de DVC, rangé avec le run MLflow
    MlflowClient(mlflow_uri).log_dict(run_id, results, str(METRICS_JSON))
    return float(results[cfg.primary_metric])


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def register_model(run_id: str, mlflow_uri: str) -> str:
    """python -m src.promote --run-id : @challenger toujours, @champion si métrique ≥ seuil."""
    from mlflow import MlflowClient

    from pipelines.stage import run_stage
    from src.promote import MODEL_NAME

    run_stage("src.promote", args=["--run-id", run_id], env={"MLFLOW_TRACKING_URI": mlflow_uri})
    versions = MlflowClient(mlflow_uri).search_model_versions(
        f"name = '{MODEL_NAME}' and run_id = '{run_id}'"
    )
    return str(versions[0].version)
