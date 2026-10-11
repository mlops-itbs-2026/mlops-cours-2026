#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 4 (CI) — Packaging du modèle
# Pré-requis : labs 0 et 1 rejoués (cluster + registry localhost:5001, MLflow avec
# scoring-model@champion, data/processed/test.csv), port-forwards actifs (make pf).
# Lance ensuite labs/m4-packaging/check.sh ; code retour = celui du check.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

# shellcheck disable=SC1091
source infra/versions.env
ENV_FILE=.env
[ -f "$ENV_FILE" ] || ENV_FILE=.env.example
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
export BENTOML_DO_NOT_TRACK=true

PORT=${E2E_PORT:-8000}
REG="localhost:${REGISTRY_PORT}"
TRIVY_IMAGE="aquasec/trivy:${TRIVY_VERSION#v}"
step() { echo; echo "=== $* ==="; }

wait_health() {
  for _ in $(seq 1 30); do
    curl -fsS "http://localhost:${PORT}/health" && echo && return 0
    sleep 1
  done
  echo "API injoignable sur le port ${PORT}" >&2
  return 1
}
predict() {
  uv run python packaging/sample_request.py -n 2 |
    curl -fsS -X POST "http://localhost:${PORT}/predict" \
      -H "Content-Type: application/json" -d @-
  echo
}
trivy() {
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    -v trivy-cache:/root/.cache/ "$TRIVY_IMAGE" image --severity HIGH,CRITICAL "$1"
}

step "0. Environnement Python"
uv sync
mkdir -p reports

step "1. Export ONNX + vérification"
uv run python packaging/app/export_onnx.py
uv run python packaging/app/verify_onnx.py
MODEL_VERSION=$(uv run python -c \
  "import json; print(json.load(open('packaging/model/metadata.json'))['model_version'])")

step "2. API FastAPI locale"
uv run python -m uvicorn app.serve:app --app-dir packaging --port "$PORT" &
API_PID=$!
wait_health
predict
kill "$API_PID"
wait "$API_PID" 2>/dev/null || true

step "3. Image naïve"
docker build -f packaging/Dockerfile.naive -t scoring-api:naive .
docker images scoring-api:naive --format "{{.Size}}"

step "4. Image multi-stage"
docker build -f packaging/Dockerfile -t scoring-api:optimized \
  --build-arg VERSION=1.0.0 --build-arg MODEL_VERSION="$MODEL_VERSION" .
docker images scoring-api:optimized --format "{{.Size}}"
docker run --rm scoring-api:optimized id
docker run -d --rm -p "${PORT}:8000" --name scoring scoring-api:optimized
wait_health
predict

step "5. Benchmark (+ Locust court)"
uv run python packaging/app/benchmark.py
uv run locust -f packaging/locustfile.py --host "http://localhost:${PORT}" \
  --users 20 --spawn-rate 10 --run-time 10s --headless --only-summary
docker stop scoring

step "6. BentoML"
uv run python packaging/bento/save_model.py
(cd packaging/bento && uv run bentoml build)
uv run bentoml containerize scoring_bento:latest -t scoring-api:bentoml
docker images scoring-api --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

step "7. Scans Trivy + ModelScan"
trivy scoring-api:optimized >reports/trivy_optimized.txt
trivy scoring-api:naive >reports/trivy_naive.txt
trivy scoring-api:bentoml >reports/trivy_bentoml.txt
grep -h "^Total" reports/trivy_*.txt || true
uv run modelscan -p packaging/model/model.onnx >reports/modelscan_onnx.txt || true
uv run python packaging/scan-demo/make_pickles.py
uv run modelscan -p packaging/scan-demo/ >reports/modelscan_pickle.txt || true
grep -E "No issues|CRITICAL" reports/modelscan_onnx.txt reports/modelscan_pickle.txt

step "8. Registry local (créé par make cluster)"
curl -fsS "http://${REG}/v2/_catalog" && echo
docker tag scoring-api:optimized "${REG}/scoring-api:1.0.0"
docker push "${REG}/scoring-api:1.0.0"
curl -fsS "http://${REG}/v2/scoring-api/tags/list" && echo
docker rmi scoring-api:optimized "${REG}/scoring-api:1.0.0"
docker pull "${REG}/scoring-api:1.0.0"
docker run -d --rm -p "${PORT}:8000" --name scoring-from-reg "${REG}/scoring-api:1.0.0"
wait_health
docker stop scoring-from-reg

step "9. Rapport"
test -s reports/rapport-m4.md && grep '^## ' reports/rapport-m4.md

step "Vérification automatique"
bash labs/m4-packaging/check.sh
