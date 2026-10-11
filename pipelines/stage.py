# Fichier : pipelines/stage.py
"""Exécute une étape du projet (python -m src.<étape>) pour un orchestrateur.

Une étape = une stage de dvc.yaml : des entrées (deps), une commande, des sorties (outs).
Elle tourne dans un dossier de travail temporaire qui reproduit l'arborescence du projet
(params.yaml, data/, models/, reports/) : le code du Lab 1, écrit avec des chemins
relatifs, s'exécute SANS modification, que l'orchestrateur soit DVC, Kubeflow Pipelines
ou Airflow. Ce module est copié dans l'image du pipeline (pipelines/Dockerfile).

- Kubeflow (pipelines/components.py) : run_stage() avec les chemins des artefacts typés.
- Airflow (dags/scoring_dag.py) : python -m pipelines.stage src.<étape> --workspace s3://…
  recopie l'arborescence du run depuis/vers S3 ; --xcom publie les petites valeurs utiles
  aux tasks suivantes (run MLflow, métriques).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # racine du projet (/app dans l'image)
OUTPUT_DIRS = ("data", "models", "reports")  # là où les étapes écrivent leurs sorties
XCOM_FILE = Path("/airflow/xcom/return.json")  # lu par le KubernetesPodOperator


def copy(src: str | Path, dst: str | Path) -> None:
    """Copie un fichier ou un dossier, en créant les dossiers parents."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copyfile(src, dst)


def execute(module: str, args, workdir: Path, env: dict | None = None) -> None:
    """Lance python -m <module> <args> dans workdir, avec le code et le contrat du projet."""
    shutil.copyfile(ROOT / "params.yaml", workdir / "params.yaml")
    run_env = {**os.environ, **(env or {})}
    run_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.getenv("PYTHONPATH")]))
    config = Path(os.getenv("DATASET_CONFIG", ROOT / "dataset.yaml")).resolve()
    run_env["DATASET_CONFIG"] = str(config)
    cmd = [sys.executable, "-m", module, *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=workdir, env=run_env, check=True)


def run_stage(module: str, args=(), inputs=None, outputs=None, env=None) -> None:
    """Copie les entrées, lance l'étape, copie les sorties (composants KFP).

    inputs  : {chemin dans le projet : fichier ou dossier à y copier avant l'étape}
    outputs : {chemin dans le projet : destination où le recopier après l'étape}
    env     : variables d'environnement en plus (ex. LAUNCHED_BY=kfp)
    """
    with tempfile.TemporaryDirectory(prefix="stage-") as tmp:
        work = Path(tmp)
        for rel, src in (inputs or {}).items():
            copy(src, work / rel)
        execute(module, args, work, env)
        for rel, dst in (outputs or {}).items():
            copy(work / rel, dst)


# --- Airflow : dossier de travail du run, sur S3 (ou en local pour tester) ---------------
def _s3(workspace: str):
    """Client S3 (endpoint et identifiants lus dans l'environnement), bucket, préfixe."""
    import boto3

    bucket, _, prefix = workspace.removeprefix("s3://").partition("/")
    prefix = re.sub(r"[^A-Za-z0-9._/-]", "_", prefix).strip("/")  # run_id Airflow : « : + »
    return boto3.client("s3"), bucket, prefix


def pull(workspace: str, work: Path) -> None:
    """Récupère les sorties des étapes précédentes du run."""
    if not workspace.startswith("s3://"):
        if Path(workspace).is_dir():
            copy(workspace, work)
        return
    s3, bucket, prefix = _s3(workspace)
    if bucket not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
        s3.create_bucket(Bucket=bucket)  # premier run : le bucket n'existe pas encore
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            target = work / obj["Key"].removeprefix(f"{prefix}/")
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(target))


def snapshot(work: Path) -> dict[Path, int]:
    """Fichiers de sortie présents et leur date de modification."""
    return {
        p: p.stat().st_mtime_ns for d in OUTPUT_DIRS for p in (work / d).rglob("*") if p.is_file()
    }


def push(work: Path, workspace: str, before: dict[Path, int]) -> None:
    """Publie les fichiers créés ou modifiés par l'étape."""
    changed = [p for p, mtime in snapshot(work).items() if before.get(p) != mtime]
    if workspace.startswith("s3://"):
        s3, bucket, prefix = _s3(workspace)
        for p in changed:
            s3.upload_file(str(p), bucket, f"{prefix}/{p.relative_to(work).as_posix()}")
    else:
        for p in changed:
            copy(p, Path(workspace) / p.relative_to(work))
    print(f"{len(changed)} fichier(s) publié(s) dans {workspace}", flush=True)


def xcom_summary(work: Path) -> dict:
    """Petites valeurs pour les tasks suivantes : run MLflow du modèle, métriques."""
    from mlflow.models import Model

    from src.evaluate import METRICS_JSON
    from src.train import MODEL_DIR

    summary = {}
    if (work / MODEL_DIR / "MLmodel").exists():
        summary["mlflow_run_id"] = Model.load(work / MODEL_DIR).metadata["mlflow_run_id"]
    if (work / METRICS_JSON).exists():
        summary["metrics"] = json.loads((work / METRICS_JSON).read_text())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exécute une étape du projet (task Airflow)", allow_abbrev=False
    )
    parser.add_argument("module", help="étape à lancer, ex. src.prepare")
    parser.add_argument("--workspace", required=True, help="s3://bucket/préfixe ou dossier")
    parser.add_argument("--xcom", action="store_true", help=f"résumé écrit dans {XCOM_FILE}")
    opts, args = parser.parse_known_args()
    args = [a for a in args if a != "--"]  # arguments de l'étape, après « -- »

    with tempfile.TemporaryDirectory(prefix="stage-") as tmp:
        work = Path(tmp)
        pull(opts.workspace, work)
        before = snapshot(work)
        execute(opts.module, args, work)
        push(work, opts.workspace, before)
        if opts.xcom:
            summary = xcom_summary(work)
            XCOM_FILE.parent.mkdir(parents=True, exist_ok=True)
            XCOM_FILE.write_text(json.dumps(summary), encoding="utf-8")
            print("XCom :", summary, flush=True)


if __name__ == "__main__":
    main()
