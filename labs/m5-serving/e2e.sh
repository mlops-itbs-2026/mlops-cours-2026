#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 5 (CI) — Serving avec KServe
# Pré-requis : labs 0, 1 et 4 rejoués sur ce runner (cluster + registry localhost:5001,
# MLflow avec scoring-model@champion, data/processed/test.csv, packaging/model/, image
# localhost:5001/scoring-api:1.0.0 poussée), port-forwards actifs (make pf).
# Options : E2E_RUN_TIME (durée de chaque run Locust, défaut 30s), M5_WITH_TRITON=1
#   (étape 8 Triton : image ≈ 6 Go), M5_SKIP_SCALE_TO_ZERO=1 (gagne ≈ 3 min).
# Lance ensuite labs/m5-serving/check.sh ; code retour = celui du check.
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

REG="localhost:${REGISTRY_PORT}"
NS=scoring
GW=http://localhost:30080
RUN_TIME=${E2E_RUN_TIME:-30s}
SEL="serving.kserve.io/inferenceservice=scoring-model"
step() { echo; echo "=== $* ==="; }
skip() { echo "SKIP ($*)"; }
isvc_host() {
  kubectl get isvc "$1" -n "$NS" -o jsonpath='{.status.url}' | sed -E 's#^[a-z]+://##; s#/.*$##'
}
wait_isvc() { kubectl wait --for=condition=Ready "isvc/$1" -n "$NS" --timeout="${2:-300s}"; }
load() { # load <nom> <utilisateurs> <durée> : run Locust headless (les erreurs n'arrêtent pas la CI)
  uv run locust -f serving/locustfile.py --host "$GW" --users "$2" \
    --spawn-rate "$(($2 / 5 > 1 ? $2 / 5 : 1))" --run-time "$3" --headless --only-summary \
    --csv "reports/m5/$1" --html "reports/m5/$1.html" || true
}

step "0. Environnement Python"
uv sync
mkdir -p reports/m5

step "1. Knative + Kourier + cert-manager + KServe + metrics-server"
make kserve
kubectl get crd | grep serving.kserve.io
kubectl get clusterservingruntimes

step "2. Image du Lab 4 et namespace"
curl -fsS "http://${REG}/v2/scoring-api/tags/list" | grep -q '"1.0.0"'
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

step "3. InferenceService minimal"
kubectl apply -f serving/scoring-isvc-v1.yaml
wait_isvc scoring-model
kubectl get isvc,ksvc,revisions,kpa -n "$NS"
ISVC_HOST=$(isvc_host scoring-model)
export ISVC_HOST
echo "Host : ${ISVC_HOST}"

step "4. Appels via la passerelle Kourier"
curl -fsS -H "Host: ${ISVC_HOST}" "${GW}/health"
echo
uv run python packaging/sample_request.py -n 2 |
  curl -fsS -X POST "${GW}/predict" -H "Host: ${ISVC_HOST}" \
    -H "Content-Type: application/json" -d @-
echo
curl -fsS -H "Host: ${ISVC_HOST}" "${GW}/metadata"
echo
kubectl run -n "$NS" client-test --rm -i --restart=Never --image=python:3.12-slim -- \
  python -c "import urllib.request; \
print(urllib.request.urlopen('http://scoring-model-predictor.scoring/health').read().decode())"

step "5. Autoscaling (KPA sur le RPS) + scale-to-zero"
kubectl apply -f serving/scoring-isvc-v1-autoscale.yaml
wait_isvc scoring-model
kubectl get kpa -n "$NS"
if [ "${M5_SKIP_SCALE_TO_ZERO:-0}" = "1" ]; then
  skip "scale-to-zero : M5_SKIP_SCALE_TO_ZERO=1"
else
  for _ in $(seq 1 48); do # jusqu'à 4 min
    n=$(kubectl get pods -n "$NS" -l "$SEL" --no-headers 2>/dev/null | grep -c Running || true)
    [ "$n" = "0" ] && break
    sleep 5
  done
  echo "Pods en cours d'exécution : ${n}"
  curl -sS -o /dev/null -w "Démarrage à froid : HTTP %{http_code} en %{time_total} s\n" \
    -H "Host: ${ISVC_HOST}" "${GW}/health"
fi

step "6. Tests de charge Locust (100 / 500 / 1000 utilisateurs, ${RUN_TIME} chacun)"
for users in 100 500 1000; do
  load "run_${users}" "$users" "$RUN_TIME"
  kubectl get kpa -n "$NS"
done
uv run python serving/locust_report.py
BATCH_SIZE=50 load batch_50 20 20s
uv run python serving/locust_report.py "reports/m5/batch_*"

step "7. Canary v2.0.0 (90/10) puis promotion"
MODEL_VERSION=$(uv run python -c \
  "import json; print(json.load(open('packaging/model/metadata.json'))['model_version'])")
docker build -f packaging/Dockerfile -t "${REG}/scoring-api:2.0.0" \
  --build-arg VERSION=2.0.0 --build-arg MODEL_VERSION="$MODEL_VERSION" .
docker push "${REG}/scoring-api:2.0.0"
kubectl apply -f serving/scoring-isvc-canary.yaml
wait_isvc scoring-model
kubectl get isvc scoring-model -n "$NS"
for _ in $(seq 1 200); do
  curl -s -H "Host: ${ISVC_HOST}" "${GW}/metadata" | grep -oE '"api_version":"[^"]+"' || true
done | sort | uniq -c
curl -fsS -H "Host: latest-${ISVC_HOST}" "${GW}/version"
echo
kubectl patch isvc scoring-model -n "$NS" --type merge \
  -p '{"spec":{"predictor":{"canaryTrafficPercent":100}}}'
wait_isvc scoring-model
kubectl get isvc scoring-model -n "$NS"

step "8. (Optionnel) BentoML et Triton dans KServe"
load compare_fastapi 20 60s
if docker image inspect scoring-api:bentoml >/dev/null 2>&1; then
  docker tag scoring-api:bentoml "${REG}/scoring-bento:1.0.0"
  docker push "${REG}/scoring-bento:1.0.0"
  kubectl apply -f serving/scoring-bento-isvc.yaml
  wait_isvc scoring-bento
  ISVC_HOST=$(isvc_host scoring-bento) load compare_bento 20 60s
  kubectl delete -f serving/scoring-bento-isvc.yaml
else
  skip "BentoML : image scoring-api:bentoml absente (Lab 4, étape 6)"
fi
if [ "${M5_WITH_TRITON:-0}" = "1" ]; then
  uv run python serving/prepare_triton.py
  kubectl create secret generic s3-credentials -n "$NS" \
    --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl annotate secret s3-credentials -n "$NS" --overwrite \
    serving.kserve.io/s3-endpoint=s3.storage:8333 \
    serving.kserve.io/s3-usehttps=0 \
    serving.kserve.io/s3-region="$AWS_DEFAULT_REGION"
  kubectl apply -f serving/kserve-s3-sa.yaml
  kubectl apply -f serving/scoring-triton-isvc.yaml
  wait_isvc scoring-triton 1800s
  TRITON_HOST=$(isvc_host scoring-triton)
  uv run python serving/v2_client.py --host "$TRITON_HOST" -n 5 --compare "$ISVC_HOST"
  ISVC_HOST=$TRITON_HOST API_FORMAT=v2 load compare_triton 20 60s
  kubectl delete -f serving/scoring-triton-isvc.yaml
else
  skip "Triton : image ≈ 6 Go ; M5_WITH_TRITON=1 pour l'activer"
fi
uv run python serving/locust_report.py "reports/m5/compare_*"

step "9. Rapport"
test -s reports/rapport-m5.md || cp labs/m5-serving/starter/rapport-m5.md reports/rapport-m5.md
grep '^## ' reports/rapport-m5.md

step "Vérification automatique"
bash labs/m5-serving/check.sh
