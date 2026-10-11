#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 0 (CI hebdomadaire) — sans intervention.
# Lancer depuis la racine du repo (branche solution/m0) : ./labs/m0-setup/e2e.sh
# Requiert sur la machine : docker, kubectl, helm, kind, uv, curl (k9s optionnel).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
# shellcheck disable=SC1091
source infra/versions.env
export DVC_NO_ANALYTICS=1 MLFLOW_DISABLE_AGENT_HINT=1

log() { printf '\n=== %s\n' "$*"; }
skip() { printf 'SKIP (%s)\n' "$*"; }

# Client S3 en ligne de commande (conteneur aws-cli, comme dans l'énoncé)
aws_lab() {
  docker run --rm --network host \
    -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
    "${AWS_CLI_IMAGE}" --endpoint-url "${AWS_ENDPOINT_URL}" "$@"
}

wait_http() {  # wait_http URL [essais]
  local url=$1 tries=${2:-60}
  for _ in $(seq 1 "$tries"); do
    if curl -s -o /dev/null -m 5 "$url"; then return 0; fi
    sleep 2
  done
  echo "Timeout en attendant ${url}" >&2
  return 1
}

log "Étapes 1-2 — outils"
for tool in docker kubectl helm kind uv curl; do
  command -v "$tool" >/dev/null || { echo "Outil manquant : ${tool}" >&2; exit 1; }
done
command -v k9s >/dev/null || skip "k9s absent : outil interactif, l'item k9s de check.sh échouera"
docker run --rm hello-world | grep "Hello from Docker"
docker compose version
kubectl version --client
helm version --short
kind version

log "Étape 3 — repo du cours"
skip "fork + clone GitHub : la CI travaille déjà dans une copie du repo"
[ -f .env ] || cp .env.example .env
set -a
# shellcheck disable=SC1091
source .env
set +a

log "Étapes 4-5 — cluster kind + registry, SeaweedFS, MLflow"
make up
kubectl --context "kind-${KIND_CLUSTER_NAME}" get nodes
curl -fsS "http://localhost:${REGISTRY_PORT}/v2/_catalog"
echo
make pf
wait_http "${MLFLOW_TRACKING_URI}/health"
wait_http "${AWS_ENDPOINT_URL}"
skip "filer UI http://localhost:8888 : pas de navigateur en CI"
aws_lab s3api head-bucket --bucket mlops-test 2>/dev/null || aws_lab s3 mb s3://mlops-test
aws_lab s3 ls

log "Étape 6 — Python 3.12 + uv"
uv python install "${PYTHON_VERSION}"
uv sync
uv run python -c "import sklearn, mlflow, dvc; print('OK', sklearn.__version__, mlflow.__version__)"
uv run dvc --version

log "Étape 7.1 — smoke test MLflow"
uv run python labs/m0-setup/smoke_mlflow.py
skip "UI MLflow http://localhost:5000 : pas de navigateur en CI"

log "Étape 7.2 — smoke test DVC (dossier jetable /tmp/dvc-smoke)"
DVC="${REPO}/.venv/bin/dvc"
GIT_ID=(-c user.name="CI Lab 0" -c user.email="ci@example.invalid")
rm -rf /tmp/dvc-smoke
mkdir -p /tmp/dvc-smoke
cd /tmp/dvc-smoke
git init -q
"$DVC" init -q
git "${GIT_ID[@]}" commit -qm "dvc init"
"$DVC" remote add -d storage s3://mlops-test/dvcstore
"$DVC" remote modify storage endpointurl "${AWS_ENDPOINT_URL}"
"$DVC" remote modify --local storage access_key_id "${AWS_ACCESS_KEY_ID}"
"$DVC" remote modify --local storage secret_access_key "${AWS_SECRET_ACCESS_KEY}"
mkdir -p data
echo "hello mlops" > data/sample.txt
"$DVC" add data/sample.txt
git add data/.gitignore data/sample.txt.dvc .dvc/config
git "${GIT_ID[@]}" commit -qm "Track sample.txt with DVC"
"$DVC" push
cd "$REPO"
aws_lab s3 ls --recursive s3://mlops-test/dvcstore/ | grep "files/md5/"

log "Étape 8 — commit/push + checklist"
skip "branche lab/m0-setup + pull request : pas de push depuis la CI"
bash labs/m0-setup/check.sh
