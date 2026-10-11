#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 2 (CI hebdomadaire) — sans intervention.
# Lancer depuis la racine du repo (branche solution/m2) : ./labs/m2-pipelines/e2e.sh
# Pré-requis : Labs 0 et 1 rejoués (cluster, SeaweedFS, MLflow, données DVC, registry) ;
# docker (avec compose), kind, kubectl, uv, git, curl.
# Les fichiers du lab (src/validate_data.py, pipelines/, dags/, rapport) sont déjà dans le repo.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export MLFLOW_DISABLE_AGENT_HINT=1 DVC_NO_ANALYTICS=1
KFP_PORT=${KFP_PORT:-8080} # port local du port-forward de l'UI/API KFP
export KFP_HOST="http://localhost:${KFP_PORT}" # lu aussi par check.sh
IMAGE=localhost:5001/scoring-pipeline:1.0.0
AF=(docker compose --project-directory infra/airflow exec -T airflow-scheduler airflow)
GIT_ID=(-c user.name="CI Lab 2" -c user.email="ci@example.invalid")
PF_PID=""
cleanup() { [ -n "$PF_PID" ] && kill "$PF_PID" 2>/dev/null || true; }
trap cleanup EXIT

log() { printf '\n=== %s\n' "$*"; }
skip() { printf 'SKIP (%s)\n' "$*"; }

wait_http() { # wait_http URL [essais]
  local url=$1 tries=${2:-60}
  for _ in $(seq 1 "$tries"); do
    if curl -fs -o /dev/null -m 5 "$url"; then return 0; fi
    sleep 5
  done
  echo "Timeout en attendant ${url}" >&2
  return 1
}

log "Préalable — port-forwards, .env, venv, données du Lab 1"
[ -f .env ] || cp .env.example .env
set -a
# shellcheck disable=SC1091
source .env
set +a
make pf
wait_http "${MLFLOW_TRACKING_URI}/health"
uv sync
# shellcheck disable=SC1091
source .venv/bin/activate
[ -f data/raw/dataset.csv ] || dvc pull

log "Étape 1 — Kubeflow Pipelines"
make kfp
kubectl -n kubeflow port-forward svc/ml-pipeline-ui "${KFP_PORT}:80" >/tmp/m2-kfp-pf.log 2>&1 &
PF_PID=$!
wait_http "${KFP_HOST}/apis/v2beta1/healthz"
curl -s "${KFP_HOST}/apis/v2beta1/healthz"
echo
python -c "import kfp; print(kfp.__version__)"
skip "UI KFP : pas de navigateur en CI"

log "Étape 2 — validation, stage, image du pipeline"
python -m src.validate_data
python - <<'EOF'
import pandas as pd

from src.dataset_config import load_config

cfg = load_config()
df = pd.read_csv(cfg.raw_path)
df.drop(columns=cfg.numeric[:1]).to_csv("/tmp/dataset_bad.csv", index=False)
EOF
if python -m src.validate_data /tmp/dataset_bad.csv; then
  echo "Le gate de validation aurait dû refuser /tmp/dataset_bad.csv" >&2
  exit 1
fi
python - <<'EOF'
from pipelines.stage import run_stage
from src.dataset_config import load_config
from src.validate_data import REPORT_JSON

cfg = load_config()
run_stage(
    "src.validate_data",
    inputs={cfg.raw_path: cfg.raw_path},
    outputs={str(REPORT_JSON): "/tmp/validation.json"},
)
EOF
docker build -f pipelines/Dockerfile --build-arg VERSION=1.0.0 -t "${IMAGE}" .
docker push "${IMAGE}"
curl -fsS http://localhost:5001/v2/scoring-pipeline/tags/list | grep -q '"1.0.0"'
docker run --rm "${IMAGE}" python -c \
  "from src.dataset_config import load_config; c = load_config(); print(c.name, len(c.features))"
docker images "${IMAGE%:*}"

log "Étape 3 — compilation et test local (kfp.local)"
python -m pipelines.scoring_pipeline
grep -c "image: ${IMAGE}" scoring_pipeline.yaml
python - <<EOF
from kfp import local

from pipelines.scoring_pipeline import scoring_pipeline

local.init(runner=local.SubprocessRunner(use_venv=False))
scoring_pipeline(mlflow_uri="${MLFLOW_TRACKING_URI}")
EOF

log "Étapes 4 à 6 — upload, deux runs identiques (cache) sur le cluster"
skip "upload par l'UI : remplacé par le SDK (même API)"
python - "${KFP_HOST}" <<'EOF'
import sys
import time

import kfp

client = kfp.Client(host=sys.argv[1])
name = "scoring-pipeline"
pipeline_id = client.get_pipeline_id(name)
if pipeline_id is None:
    pipeline_id = client.upload_pipeline("scoring_pipeline.yaml", pipeline_name=name).pipeline_id
    versions = client.list_pipeline_versions(pipeline_id).pipeline_versions
    version_id = versions[0].pipeline_version_id
else:
    version_id = client.upload_pipeline_version(
        "scoring_pipeline.yaml", pipeline_version_name=f"e2e-{int(time.time())}",
        pipeline_id=pipeline_id,
    ).pipeline_version_id
experiment = client.create_experiment("scoring")
for i in (1, 2):  # le 2e run doit servir ingest/validate/prepare depuis le cache
    run = client.run_pipeline(experiment.experiment_id, f"scoring-e2e-{i}",
                              pipeline_id=pipeline_id, version_id=version_id)
    state = client.wait_for_run_completion(run.run_id, timeout=1800).state
    print(f"run {i} : {run.run_id} -> {state}")
    assert state == "SUCCEEDED", state
EOF
skip "inspection visuelle du cache et de la matrice de confusion : UI"

log "Étape 5 — MLflow vu depuis un pod, registry"
kubectl -n mlflow get pods,svc
kubectl -n kubeflow delete pod mlflow-check --ignore-not-found
kubectl -n kubeflow run mlflow-check --rm -i --restart=Never --image="${IMAGE}" -- python -c \
  "import urllib.request as u; print(u.urlopen('http://mlflow.mlflow:5000/health').read())"
python - <<'EOF'
from mlflow import MlflowClient

client = MlflowClient()
for alias in ("challenger", "champion"):
    mv = client.get_model_version_by_alias("scoring-model", alias)
    tags = client.get_run(mv.run_id).data.tags
    print(f"@{alias}: version {mv.version}, launched_by={tags['launched_by']}, "
          f"kfp_run_id={tags.get('kfp_run_id')}")
EOF

log "Étape 6 — politiques dans le YAML compilé"
grep -nE "maxRetryCount|resourceMemoryLimit|activeDeadlineSeconds" scoring_pipeline.yaml

log "Étape 7 — Airflow (Docker Compose) et DAG scoring_dag"
make airflow
for _ in $(seq 1 30); do # le dag-processor scanne dags/ toutes les 30 s
  if "${AF[@]}" dags list -o json 2>/dev/null | grep -q '"scoring_dag"'; then break; fi
  sleep 10
done
"${AF[@]}" dags list-import-errors
"${AF[@]}" dags unpause scoring_dag
RUN_ID="e2e__$(date -u +%Y%m%dT%H%M%S)"
"${AF[@]}" dags trigger scoring_dag --run-id "${RUN_ID}"
state=""
for _ in $(seq 1 120); do # jusqu'à 20 min
  state=$("${AF[@]}" dags list-runs scoring_dag -o json 2>/dev/null | tail -n 1 |
    python -c "import json,sys; r={x['run_id']: x['state'] for x in json.load(sys.stdin)}; \
print(r.get('${RUN_ID}', ''))" || true)
  if [ "$state" = "success" ] || [ "$state" = "failed" ]; then break; fi
  sleep 10
done
echo "Run Airflow : ${state}"
[ "$state" = "success" ]
skip "UI Airflow (Graph, XCom) : pas de navigateur en CI"

log "Étape 8 — comparaison"
docker stats --no-stream
wc -l pipelines/components.py pipelines/scoring_pipeline.py dags/scoring_dag.py
rm -rf /tmp/ws
python -m pipelines.stage src.download --workspace /tmp/ws
python -m pipelines.stage src.validate_data --workspace /tmp/ws
ls -R /tmp/ws
skip "mesures de DX et tests manuels (hot-reload, panne volontaire) : rédigés dans le rapport"

log "Étape 9 — rapport et commit"
skip "rédaction : reports/rapport-m2.md est fourni par la solution"
git add src/validate_data.py pipelines/ dags/ reports/rapport-m2.md
git status --short
git "${GIT_ID[@]}" commit -qm "feat: pipeline KFP scoring-pipeline + DAG Airflow scoring_dag" || true
skip "branche lab/m2-pipelines + pull request : pas de push vers GitHub depuis la CI"

log "Vérification automatique"
bash labs/m2-pipelines/check.sh
