#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 3 (CI hebdomadaire) — sans intervention.
# Lancer depuis la racine du repo (branche solution/m3) : ./labs/m3-validation/e2e.sh
# Pré-requis : Labs 0 à 2 rejoués (cluster, SeaweedFS, MLflow, KFP, données DVC, modèle,
# image 1.0.0) ; docker, kind, kubectl, uv, git, curl.
# Les fichiers du lab (gx/, src/, tests/, pipelines/, params.yaml…) sont déjà dans le repo.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export MLFLOW_DISABLE_AGENT_HINT=1 DVC_NO_ANALYTICS=1 DO_NOT_TRACK=true
KFP_PORT=${KFP_PORT:-8080} # port local du port-forward de l'UI/API KFP
export KFP_HOST="http://localhost:${KFP_PORT}" # lu aussi par check.sh
GIT_ID=(-c user.name="CI Lab 3" -c user.email="ci@example.invalid")
PF_PIDS=()
cleanup() { for p in "${PF_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

log() { printf '\n=== %s\n' "$*"; }
skip() { printf 'SKIP (%s)\n' "$*"; }
must_fail() { # must_fail commande… : la commande DOIT échouer (gate)
  if "$@"; then
    echo "ÉCHEC DU REJEU : « $* » aurait dû échouer" >&2
    exit 1
  fi
}

wait_http() { # wait_http URL [essais]
  local url=$1 tries=${2:-60}
  for _ in $(seq 1 "$tries"); do
    if curl -fs -o /dev/null -m 5 "$url"; then return 0; fi
    sleep 5
  done
  echo "Timeout en attendant ${url}" >&2
  return 1
}

# Upload (ou nouvelle version) d'un pipeline compilé, run, attente et état attendu
kfp_run() { # kfp_run fichier.yaml nom_du_pipeline ÉTAT_ATTENDU
  python - "$KFP_HOST" "$1" "$2" "$3" <<'EOF'
import sys
import time

import kfp

host, package, name, expected = sys.argv[1:5]
client = kfp.Client(host=host)
pipeline_id = client.get_pipeline_id(name)
if pipeline_id is None:
    pipeline_id = client.upload_pipeline(package, pipeline_name=name).pipeline_id
    versions = client.list_pipeline_versions(pipeline_id).pipeline_versions
    version_id = versions[0].pipeline_version_id
else:
    version_id = client.upload_pipeline_version(
        package, pipeline_version_name=f"e2e-{int(time.time())}", pipeline_id=pipeline_id
    ).pipeline_version_id
experiment = client.create_experiment("scoring")
run = client.run_pipeline(experiment.experiment_id, f"{name}-e2e", pipeline_id=pipeline_id,
                          version_id=version_id)
state = client.wait_for_run_completion(run.run_id, timeout=1800).state
print(f"{name} : run {run.run_id} -> {state}")
assert state == expected, f"attendu {expected}"
with open("/tmp/m3-last-run", "w") as f:
    f.write(run.run_id)
EOF
}

log "Préalable — port-forwards, .env, venv, données et modèle du Lab 1"
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
if [ ! -f data/raw/dataset.csv ] || [ ! -f models/model/MLmodel ]; then dvc pull; fi
kubectl -n kubeflow port-forward svc/ml-pipeline-ui "${KFP_PORT}:80" >/tmp/m3-kfp-pf.log 2>&1 &
PF_PIDS+=($!)
wait_http "${KFP_HOST}/apis/v2beta1/healthz"

log "Étape 1 — suite Great Expectations"
python -c "import great_expectations as gx; print(gx.__version__)"
# La suite est versionnée : on vérifie que la regénérer donne les mêmes expectations
# (détecte une rupture d'API de GX), puis on restaure la version commitée.
python - <<'EOF' >/tmp/m3-suite-before.json
import json

suite = json.load(open("gx/expectations/scoring_contract.json"))["expectations"]
print(json.dumps([(e["type"], e["kwargs"]) for e in suite], sort_keys=True))
EOF
python -m src.validate_data --build-suite
python - <<'EOF' >/tmp/m3-suite-after.json
import json

suite = json.load(open("gx/expectations/scoring_contract.json"))["expectations"]
print(json.dumps([(e["type"], e["kwargs"]) for e in suite], sort_keys=True))
EOF
diff /tmp/m3-suite-before.json /tmp/m3-suite-after.json
git checkout -- gx/
python -m src.validate_data
python - <<'EOF'
import pandas as pd

from src.dataset_config import load_config

cfg = load_config()
df = pd.read_csv(cfg.raw_path)
df.loc[df.index[:20], cfg.numeric[0]] *= -1  # 20 valeurs négatives (duration)
df.to_csv("/tmp/dataset_bad.csv", index=False)
EOF
must_fail python -m src.validate_data /tmp/dataset_bad.csv
python -m src.simulate corrupt data/raw/dataset.csv /tmp/dataset_corrupted.csv
must_fail python -m src.validate_data /tmp/dataset_corrupted.csv
skip "Data Docs et rapports HTML : pas de navigateur en CI"
python - <<'EOF'
import pandas as pd
import pandera.pandas as pa

from src.dataset_config import load_config

cfg = load_config()
ref = pd.read_csv("data/processed/train.csv")
schema = pa.DataFrameSchema(
    {c: pa.Column(float, pa.Check.ge(min(0, ref[c].min())), coerce=True) for c in cfg.numeric}
    | {c: pa.Column(str, pa.Check.isin(ref[c].unique())) for c in cfg.categorical}
    | {cfg.target: pa.Column(str, pa.Check.isin(ref[cfg.target].unique()))}
)
schema.validate(pd.read_csv(cfg.raw_path))
print("Données brutes : conformes")
try:
    schema.validate(pd.read_csv("/tmp/dataset_bad.csv"), lazy=True)
except pa.errors.SchemaErrors as err:
    print(err.failure_cases[["column", "check", "failure_case"]].head(3))
EOF

log "Étape 2 — image du pipeline"
skip "image intermédiaire 1.1.0 : la solution construit directement le tag final"
IMAGE=$(python -c "from pipelines.components import PIPELINE_IMAGE as i; print(i)")
grep -q "IMAGE = \"${IMAGE}\"" dags/scoring_dag.py # même image pour KFP et Airflow
docker build -f pipelines/Dockerfile --build-arg VERSION="${IMAGE##*:}" -t "${IMAGE}" .
docker push "${IMAGE}"
docker run --rm "${IMAGE}" python -c \
  "import great_expectations as gx, evidently; print(gx.__version__, evidently.__version__)"
docker run --rm "${IMAGE}" ls gx

log "Étapes 3 et 6.3 — compilation et test local du pipeline à 3 gates (kfp.local)"
python -m pipelines.scoring_pipeline
grep -c "image: ${IMAGE}" scoring_pipeline.yaml
python - <<EOF
from kfp import local

from pipelines.scoring_pipeline import scoring_pipeline

local.init(runner=local.SubprocessRunner(use_venv=False))
scoring_pipeline(mlflow_uri="${MLFLOW_TRACKING_URI}")
EOF

log "Étape 4 — données corrompues : le gate doit bloquer"
python -m pipelines.test_bad_data
python - <<'EOF'
from kfp import local

from pipelines.test_bad_data import bad_data_pipeline

local.init(runner=local.SubprocessRunner(use_venv=False), raise_on_error=False)
bad_data_pipeline()
EOF
ls local_outputs/scoring-bad-data-test-*/validate-data/html
skip "upload par l'UI : remplacé par le SDK (même API)"
kfp_run test_bad_data.yaml scoring-bad-data-test FAILED

log "Étape 5 — drift"
python -m src.detect_drift
python -m src.simulate drift data/processed/test.csv /tmp/test_drifted.csv
python -m src.validate_data /tmp/test_drifted.csv
must_fail python -m src.detect_drift data/processed/train.csv /tmp/test_drifted.csv
python -m pipelines.drift_demo
python - <<'EOF'
from kfp import local

from pipelines.drift_demo import drift_pipeline

local.init(runner=local.SubprocessRunner(use_venv=False))
drift_pipeline()
EOF

log "Étape 6 — validation du modèle, runs sur le cluster"
python -m pytest -q tests/test_model_behavior.py
python -m src.validate_model
kfp_run scoring_pipeline.yaml scoring-pipeline SUCCEEDED
RUN_ID=$(cat /tmp/m3-last-run)
kfp_run drift_demo.yaml scoring-drift-demo SUCCEEDED
skip "inspection visuelle des rapports HTML dans l'UI KFP"

log "Étape 7 — télécharger un rapport depuis le SeaweedFS interne de KFP"
kubectl -n kubeflow port-forward svc/seaweedfs 9090:8333 >/tmp/m3-kfp-s3.log 2>&1 &
PF_PIDS+=($!)
sleep 3
KFP_S3_KEY=$(kubectl -n kubeflow get secret mlpipeline-minio-artifact \
  -o jsonpath='{.data.accesskey}' | base64 -d)
KFP_S3_SECRET=$(kubectl -n kubeflow get secret mlpipeline-minio-artifact \
  -o jsonpath='{.data.secretkey}' | base64 -d)
kfp-s3() {
  mkdir -p reports/kfp
  docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD/reports/kfp:/data" -e AWS_ACCESS_KEY_ID="$KFP_S3_KEY" \
    -e AWS_SECRET_ACCESS_KEY="$KFP_S3_SECRET" -e AWS_DEFAULT_REGION=us-east-1 \
    amazon/aws-cli:2.37.9 --endpoint-url http://localhost:9090 "$@"
}
kfp-s3 s3 ls s3://mlpipeline/v2/artifacts/scoring-pipeline/
kfp-s3 s3 ls --recursive "s3://mlpipeline/v2/artifacts/scoring-pipeline/$RUN_ID/" | grep /html
KEY=$(kfp-s3 s3 ls --recursive "s3://mlpipeline/v2/artifacts/scoring-pipeline/$RUN_ID/" |
  awk '/detect-drift\/.*\/html$/ {print $4}')
kfp-s3 s3 cp "s3://mlpipeline/$KEY" /data/drift.html
test -s reports/kfp/drift.html

log "Commit"
git add params.yaml gx/ src/ tests/ pipelines/ dags/scoring_dag.py
git status --short
git "${GIT_ID[@]}" commit -qm "feat: gates de validation (GX, drift, modèle) dans scoring-pipeline" ||
  true
skip "branche lab/m3-validation + pull request : pas de push vers GitHub depuis la CI"

log "Vérification automatique"
bash labs/m3-validation/check.sh
