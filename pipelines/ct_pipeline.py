# Fichier : pipelines/ct_pipeline.py
"""Pipeline KFP « ct-pipeline » du Lab 6 : ré-entraînement déclenché par une dérive.

ingest (lot récent) -> validate -> prepare -> train -> evaluate      composants du Lab 2
                    -> register_challenger -> promotion_gate         nouveaux (Lab 6)

Le pipeline ne touche PAS au cluster : il produit un @champion dans MLflow, et la mise en
production passe par Git (pipelines/gitops.py), comme pour un humain.
Compilation : python -m pipelines.ct_pipeline   (écrit ct_pipeline.yaml)
Les hyper-paramètres sont ceux de params.yaml, lus à la compilation : le CT ré-entraîne
la configuration validée, sur des données plus récentes.
"""

from pathlib import Path

import yaml
from kfp import compiler, dsl, kubernetes
from kfp.dsl import Dataset, Input, Markdown, Metrics, Output

from pipelines.components import (
    MLFLOW_URI,
    PIPELINE_IMAGE,
    evaluate_model,
    prepare_features,
    train_model,
    validate_data,
)

TRAIN = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["train"]


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def ingest_ct_data(data_uri: str, mlflow_uri: str, raw_data: Output[Dataset]) -> None:
    """Le lot de données étiquetées à apprendre : data_uri (artefact MLflow), sinon la source."""
    import shutil

    import mlflow
    import pandas as pd

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import file_md5

    cfg = load_config()
    if data_uri:
        # Lot déposé avec le rapport de dérive (runs:/<id>/…) : le serveur MLflow fait
        # proxy des artefacts, le pod n'a besoin d'aucun identifiant S3.
        local = mlflow.artifacts.download_artifacts(artifact_uri=data_uri, tracking_uri=mlflow_uri)
        shutil.copyfile(local, raw_data.path)
    else:
        run_stage("src.download", outputs={cfg.raw_path: raw_data.path})
    raw_data.metadata["dataset"] = cfg.name
    raw_data.metadata["source"] = data_uri or "src.download"
    raw_data.metadata["rows"] = len(pd.read_csv(raw_data.path))
    raw_data.metadata["md5"] = file_md5(raw_data.path)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def register_challenger(
    run_id: str,
    mlflow_uri: str,
    kfp_run_id: str,
    trigger_source: str,
    drift_share: float,
    data_uri: str,
) -> str:
    """Enregistre le modèle du run comme nouvelle version, alias @challenger SEULEMENT.

    Contrairement à src.promote (Lab 1), aucun @champion ici : c'est le rôle de la gate.
    """
    import mlflow
    from mlflow import MlflowClient

    from src.promote import MODEL_NAME

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    # Pourquoi ce run existe : traçabilité du ré-entraînement (lu par make check-m6)
    context = {"ct_trigger": trigger_source, "ct_drift_share": f"{drift_share:.4f}",
               "ct_data_uri": data_uri or "src.download"}  # fmt: skip
    for key, value in context.items():
        client.set_tag(run_id, key, value)

    model_id = client.get_run(run_id).outputs.model_outputs[0].model_id  # MLflow 3
    version = mlflow.register_model(f"models:/{model_id}", MODEL_NAME).version
    for key, value in {**context, "kfp_run_id": kfp_run_id, "promotion": "pending"}.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"{MODEL_NAME} version {version} -> @challenger")
    return str(version)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def promotion_gate(
    version: str,
    test_data: Input[Dataset],
    mlflow_uri: str,
    max_cost_increase: float,
    gate_metrics: Output[Metrics],
    report: Output[Markdown],
) -> str:
    """Gate challenger -> champion, sur les MÊMES données récentes pour les deux modèles.

    1. métrique primaire du challenger >= metrics.promotion_threshold (dataset.yaml) ;
    2. coût métier du challenger <= coût du champion actuel x (1 + max_cost_increase).
    Chaque modèle décide à SON seuil (param decision_threshold de son run, Lab 1).
    Renvoie « promoted » ou « rejected » : un refus n'est pas une erreur du pipeline.
    """
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from src.dataset_config import load_config
    from src.promote import MODEL_NAME
    from src.train import compute_metrics

    cfg = load_config()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    X, y = cfg.read_xy(test_data.path)

    def evaluate(v: str) -> dict:
        run_id = client.get_model_version(MODEL_NAME, v).run_id
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{v}")
        threshold = float(client.get_run(run_id).data.params["decision_threshold"])
        return compute_metrics(cfg, y, model.predict_proba(X)[:, 1], threshold)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    except MlflowException:
        champion = None  # premier modèle : seule la règle 1 s'applique
    new = evaluate(version)
    old = evaluate(champion) if champion else None

    metric, cost = cfg.primary_metric, "business_cost"
    checks = {f"{metric} >= {cfg.promotion_threshold}": new[metric] >= cfg.promotion_threshold}
    if old is not None:
        limit = old[cost] * (1 + max_cost_increase)
        checks[f"{cost} <= {limit:g} (champion v{champion})"] = new[cost] <= limit
    passed = all(checks.values())

    for name, value in new.items():
        gate_metrics.log_metric(f"challenger_{name}", float(value))
    for name, value in (old or {}).items():
        gate_metrics.log_metric(f"champion_{name}", float(value))
    lines = [f"## Gate de promotion : version {version} {'PROMUE' if passed else 'REFUSÉE'}", "",
             "| Règle | Résultat |", "|---|---|"]  # fmt: skip
    lines += [f"| {rule} | {'OK' if ok else 'ÉCHEC'} |" for rule, ok in checks.items()]
    lines += ["", "| Modèle | " + metric + " | coût métier |", "|---|---|---|",
              f"| challenger v{version} | {new[metric]:.4f} | {new[cost]:g} |"]  # fmt: skip
    if old is not None:
        lines.append(f"| champion v{champion} | {old[metric]:.4f} | {old[cost]:g} |")
    with open(report.path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    tags = {"promotion": "passed" if passed else "rejected",
            "validation_status": "passed" if passed else "failed",
            metric: f"{new[metric]:.4f}", cost: f"{new[cost]:g}"}  # fmt: skip
    if passed:
        # Retour arrière possible (gitops.py rollback) : on note le champion remplacé
        tags["previous_champion"] = champion or "none"
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)
    return "promoted" if passed else "rejected"


@dsl.pipeline(
    name="ct-pipeline",
    description="CT : ingest -> validate -> prepare -> train -> evaluate -> challenger -> gate",
)
def ct_pipeline(
    data_uri: str = "",  # artefact MLflow du lot récent ; vide = source (src.download)
    trigger_source: str = "manual",  # evidently-drift, alertmanager, manual…
    drift_share: float = 0.0,  # part des variables en dérive (rapport Evidently)
    max_cost_increase: float = 0.0,  # tolérance de la gate sur le coût métier (0 = aucune)
    mlflow_uri: str = MLFLOW_URI,
):
    ingest_task = ingest_ct_data(data_uri=data_uri, mlflow_uri=mlflow_uri)
    ingest_task.set_retry(num_retries=3, backoff_duration="30s")
    ingest_task.set_cpu_limit("500m").set_memory_limit("512Mi")
    kubernetes.set_timeout(ingest_task, 600)
    ingest_task.set_caching_options(enable_caching=False)  # le lot change à chaque dérive

    # Lab 2 (et Lab 3) : la même gate de données que le pipeline d'entraînement
    validate_task = validate_data(raw_data=ingest_task.outputs["raw_data"])
    validate_task.set_memory_limit("512Mi")

    prep_task = prepare_features(raw_data=ingest_task.outputs["raw_data"])
    prep_task.after(validate_task)
    prep_task.set_memory_limit("1Gi")

    train_task = train_model(
        raw_data=ingest_task.outputs["raw_data"],
        train_data=prep_task.outputs["train_data"],
        test_data=prep_task.outputs["test_data"],
        model_type=TRAIN["model"],
        C=TRAIN["C"],
        class_weight=str(TRAIN["class_weight"]),
        n_estimators=TRAIN["n_estimators"],
        max_depth=str(TRAIN["max_depth"]),
        learning_rate=TRAIN["learning_rate"],
        random_state=TRAIN["random_state"],
        mlflow_uri=mlflow_uri,
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    train_task.set_cpu_request("500m").set_cpu_limit("2")
    train_task.set_memory_request("512Mi").set_memory_limit("2Gi")
    kubernetes.set_timeout(train_task, 1800)
    train_task.set_caching_options(enable_caching=False)

    eval_task = evaluate_model(
        model=train_task.outputs["model"],
        test_data=prep_task.outputs["test_data"],
        run_id=train_task.outputs["Output"],
        mlflow_uri=mlflow_uri,
    )
    eval_task.set_memory_limit("1Gi")

    register_task = register_challenger(
        run_id=train_task.outputs["Output"],
        mlflow_uri=mlflow_uri,
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        trigger_source=trigger_source,
        drift_share=drift_share,
        data_uri=data_uri,
    )
    register_task.after(eval_task)  # on n'enregistre qu'un modèle évalué
    register_task.set_memory_limit("1Gi")
    register_task.set_caching_options(enable_caching=False)

    gate_task = promotion_gate(
        version=register_task.output,
        test_data=prep_task.outputs["test_data"],
        mlflow_uri=mlflow_uri,
        max_cost_increase=max_cost_increase,
    )
    gate_task.set_memory_limit("1Gi")
    gate_task.set_caching_options(enable_caching=False)


if __name__ == "__main__":
    compiler.Compiler().compile(ct_pipeline, package_path="ct_pipeline.yaml")
