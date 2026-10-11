# Fichier : dags/scoring_dag.py
"""DAG Airflow « scoring_dag » : la variante Airflow du pipeline KFP du Lab 2.

Même image (localhost:5001/scoring-pipeline), mêmes commandes que dvc.yaml et que les
composants KFP : chaque task est un pod lancé dans le cluster kind par le
KubernetesPodOperator. Différence clé avec KFP : Airflow ne transporte AUCUN fichier.
Chaque task recopie l'arborescence du projet depuis/vers un préfixe S3 propre au run
(pipelines/stage.py --workspace) ; seules de petites valeurs passent par XCom
(identifiant du run MLflow, métriques).
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG
from kubernetes.client import models as k8s

IMAGE = "localhost:5001/scoring-pipeline:1.0.0"  # même tag que pipelines/components.py
WORKSPACE = "s3://airflow-data/scoring/{{ run_id }}"  # dossier de travail du run (S3)
ENV = {
    "MLFLOW_TRACKING_URI": "http://mlflow.mlflow:5000",  # aucun identifiant S3 pour MLflow
    "LAUNCHED_BY": "airflow",  # tag launched_by des runs MLflow
    "AWS_ENDPOINT_URL": "http://s3.storage:8333",  # S3 du cours, pour le dossier de travail
    "AWS_DEFAULT_REGION": "us-east-1",
}
S3_CREDENTIALS = k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="s3-credentials"))


def stage(task_id: str, module: str, *args: str, xcom=False, memory="1Gi", retries=0):
    """Une task = un pod qui exécute « python -m pipelines.stage <module> » (une stage DVC)."""
    return KubernetesPodOperator(
        task_id=task_id,
        name=f"scoring-{task_id}",
        namespace="airflow",  # créé par make airflow, avec le secret s3-credentials
        image=IMAGE,
        cmds=["python", "-m", "pipelines.stage", module, "--workspace", WORKSPACE],
        arguments=[*(["--xcom"] if xcom else []), "--", *args],
        env_vars=ENV,
        env_from=[S3_CREDENTIALS],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"}, limits={"memory": memory}
        ),
        do_xcom_push=xcom,  # lit /airflow/xcom/return.json écrit par --xcom
        retries=retries,
        get_logs=True,
        on_finish_action="delete_succeeded_pod",
    )


with DAG(
    dag_id="scoring_dag",
    description="Credit scoring : ingest -> validate -> prepare -> train -> evaluate -> register",
    schedule=None,  # déclenchement manuel (cron possible : "0 3 * * *")
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    default_args={"owner": "mlops-student", "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "lab2"],
) as dag:
    ingest = stage("ingest", "src.download", retries=3)  # source externe : retries
    validate = stage("validate", "src.validate_data", memory="512Mi")
    prepare = stage("prepare", "src.prepare")
    train = stage("train", "src.train", xcom=True, memory="2Gi")
    evaluate = stage("evaluate", "src.evaluate", xcom=True)
    register = stage(
        "register",
        "src.promote",
        "--run-id",
        "{{ ti.xcom_pull(task_ids='train')['mlflow_run_id'] }}",
    )

    ingest >> validate >> prepare >> train >> evaluate >> register
