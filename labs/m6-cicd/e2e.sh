#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 6 (CI du cours) — CI/CD/CT
# Pré-requis : labs 0 à 5 rejoués sur ce runner (cluster + registry localhost:5001, KFP et
# image du pipeline, KServe + InferenceService scoring-model du Lab 5, MLflow avec
# scoring-model@champion, data/processed/, packaging/model/), port-forwards (make pf).
# Ce qu'un runner ne peut pas faire est remplacé, et signalé par SKIP :
#   - fork GitHub + GitHub Actions -> dépôt Git servi DANS le cluster (git daemon, lu par
#     Argo CD) et jobs de ci.yml rejoués ici (mêmes commandes ; registry local, pas GHCR) ;
#   - le bot CT construit donc l'image localement (CT_LOCAL_CI=1, gitops release --local-ci).
# Options : KFP_PORT (8080), E2E_CT_TIMEOUT (s, défaut 2400).
# Lance ensuite labs/m6-cicd/check.sh ; code retour = celui du check.
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
GW=http://localhost:30080
KFP_PORT=${KFP_PORT:-8080}
CT_TIMEOUT=${E2E_CT_TIMEOUT:-2400}
GIT_PORT=19418
IN_CLUSTER_REPO=git://git-server.gitops-e2e.svc.cluster.local/mlops-cours-2026.git
export KFP_HOST="http://localhost:${KFP_PORT}" GATEWAY="$GW"
export GITOPS_REMOTE=e2e CT_LOCAL_CI=1 CHECK_ALLOW_LOCAL_REGISTRY=1 CT_COOLDOWN_SECONDS=0
export CT_WEBHOOK_PORT=8095 MLFLOW_DISABLE_TELEMETRY=true
CT_WEBHOOK_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(16))")
export CT_WEBHOOK_TOKEN
PIDS=()
step() { echo; echo "=== $* ==="; }
skip() { echo "SKIP ($*)"; }
cleanup() { [ "${#PIDS[@]}" -eq 0 ] || kill "${PIDS[@]}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
host() { echo "${3:-}scoring-model-predictor.${1}.example.com"; } # host <ns> _ [route]
version_of() { curl -fsS --max-time 60 -H "Host: $(host "$1" _ "${2:-}")" "${GW}/version"; }
wait_version() { # wait_version <ns> <tag> [route] : jusqu'à 10 min
  for _ in $(seq 1 60); do
    version_of "$1" "${3:-}" 2>/dev/null | grep -Eq "\"version\": ?\"$2\"" && return 0
    sleep 10
  done
  echo "Délai dépassé : $1 ne sert pas $2" >&2
  return 1
}
wait_app() { # wait_app <application> : Synced + Healthy
  kubectl -n argocd wait "application/$1" --for=jsonpath='{.status.sync.status}'=Synced \
    --timeout=600s
  kubectl -n argocd wait "application/$1" --for=jsonpath='{.status.health.status}'=Healthy \
    --timeout=600s
}
gitops() { uv run python -m pipelines.gitops "$@"; }
local_ci() { gitops local-ci --push | tail -1; } # jobs build + deploy-staging, sans GitHub
start_webhook() { # start_webhook <CT_AUTO_RELEASE>
  CT_AUTO_RELEASE=$1 uv run python -m pipelines.trigger_ct serve >"/tmp/ct-webhook-$1.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 30); do
    curl -fsS "localhost:${CT_WEBHOOK_PORT}/health" && return 0
    sleep 2
  done
  return 1
}
wait_ct() { # wait_ct <run_id> <clé attendue> : suit /runs du webhook
  [ -n "$1" ] || {
    echo "Le webhook n'a pas lancé de run (voir sa réponse ci-dessus)" >&2
    return 1
  }
  local deadline=$((SECONDS + CT_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS "localhost:${CT_WEBHOOK_PORT}/runs" | python3 -c "
import json, sys
runs = [r for r in json.load(sys.stdin) if r['run_id'] == '$1']
sys.exit(0 if runs and ('$2' in runs[0] or 'error' in runs[0]) else 1)"; then
      curl -fsS "localhost:${CT_WEBHOOK_PORT}/runs"
      echo
      curl -fsS "localhost:${CT_WEBHOOK_PORT}/runs" | grep -q '"error"' && return 1
      return 0
    fi
    sleep 15
  done
  return 1
}

step "0. Environnement Python, identité Git du runner"
uv sync
git config user.name >/dev/null || git config user.name "CI du cours"
git config user.email >/dev/null || git config user.email "ci@mlops-cours.invalid"
grep -q '^CT_WEBHOOK_TOKEN=' .env 2>/dev/null ||
  echo "CT_WEBHOOK_TOKEN=${CT_WEBHOOK_TOKEN}" >>.env # comme l'étape 6.2 (.env est ignoré)
CT_WEBHOOK_TOKEN=$(sed -n 's/^CT_WEBHOOK_TOKEN=//p' .env | tail -1)
export CT_WEBHOOK_TOKEN

step "1-2. Job « test » de ci.yml : lint, tests, schéma du workflow"
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uvx check-jsonschema@0.38.2 --builtin-schema vendor.github-workflows .github/workflows/ci.yml
skip "GitHub Actions et publication sur GHCR : pas de fork ni de GITHUB_TOKEN sur ce runner"

step "2. Job « build » de ci.yml : image + test de fumée (registry local)"
TAG="$(uv run python -c 'from pipelines.gitops import ci_api_version as v; print(v())').$(git rev-list --count HEAD)"
MODEL_VERSION=$(jq -r .model_version packaging/model/metadata.json)
docker build -f packaging/Dockerfile -t "${REG}/scoring-api:${TAG}" \
  --build-arg VERSION="$TAG" --build-arg MODEL_VERSION="$MODEL_VERSION" .
docker rm -f e2e-api >/dev/null 2>&1 || true
docker run -d --name e2e-api -p 18600:8000 "${REG}/scoring-api:${TAG}"
for _ in $(seq 1 30); do curl -fsS localhost:18600/health && break; sleep 1; done
curl -fsS localhost:18600/version | grep -q "\"version\":\"${TAG}\""
gitops payload >/tmp/request.json
curl -fsS -X POST localhost:18600/predict -H "Content-Type: application/json" -d @/tmp/request.json
echo
docker rm -f e2e-api
docker push "${REG}/scoring-api:${TAG}"

step "3. Argo CD"
make argocd

step "4. Dépôt Git dans le cluster + Applications Argo CD"
kubectl apply -f labs/m6-cicd/starter/git-server.yaml
kubectl -n gitops-e2e rollout status deploy/git-server --timeout=300s
kubectl -n gitops-e2e port-forward svc/git-server "${GIT_PORT}:9418" >/dev/null 2>&1 &
PIDS+=($!)
sleep 3
git remote remove e2e 2>/dev/null || true
git remote add e2e "git://localhost:${GIT_PORT}/mlops-cours-2026.git"
git push -q e2e HEAD:main
gitops deploy --env staging --image "${REG}/scoring-api" --tag "$TAG" \
  --note "[$(git rev-parse --short HEAD)]" --push
sed "s#https://github.com/VOTRE_USER/mlops-cours-2026.git#${IN_CLUSTER_REPO}#" \
  k8s/argocd-app.yaml | kubectl apply -f -
wait_app scoring-staging
wait_app scoring-prod
kubectl get applications -n argocd
kubectl get isvc -A
wait_version scoring-staging "$TAG"

step "5. Déploiement par git push (code -> CI -> staging)"
sed -i '1a # Lab 6 : déploiement par git push (rejeu e2e)' packaging/app/serve.py
git commit -qam "Lab 6: déploiement par git push"
git push -q e2e HEAD:main
TAG2=$(local_ci)
git pull -q --rebase e2e main
wait_version scoring-staging "$TAG2"
echo "staging : $(version_of scoring-staging)"

step "6. Webhook de ré-entraînement (sans mise en prod automatique)"
if ! curl -fsS "${KFP_HOST}/apis/v2beta1/healthz" >/dev/null 2>&1; then
  kubectl -n kubeflow port-forward svc/ml-pipeline-ui "${KFP_PORT}:80" >/dev/null 2>&1 &
  PIDS+=($!)
  sleep 3
fi
start_webhook 0
WEBHOOK_PID=${PIDS[-1]}
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "localhost:${CT_WEBHOOK_PORT}/drift" \
  -H "Content-Type: application/json" -d '{"drift_share": 0.5}')
[ "$code" = "401" ] # sans jeton : refusé

step "7. Dérive simulée -> Evidently -> webhook -> pipeline CT (KFP) -> gate"
uv run python -m pipelines.trigger_ct simulate
uv run python -m pipelines.trigger_ct detect | tee /tmp/ct-detect.log
RUN_ID=$(sed -n 's/.*"run_id": "\([^"]*\)".*/\1/p' /tmp/ct-detect.log | tail -1)
wait_ct "$RUN_ID" promoted_version
uv run python -c "from mlflow import MlflowClient; c = MlflowClient(); \
mv = c.get_model_version_by_alias('scoring-model', 'champion'); print(mv.version, mv.tags)"
kill "$WEBHOOK_PID"

step "8. Mise en prod par un humain : modèle -> CI -> staging -> canary 10 % -> promotion"
git pull -q --rebase e2e main
uv run python packaging/app/export_onnx.py
git add packaging/model
git commit -qm "Lab 6: modèle ré-entraîné (étape 8)"
git push -q e2e HEAD:main
TAG3=$(local_ci)
git pull -q --rebase e2e main
wait_version scoring-staging "$TAG3"
gitops deploy --env prod --canary 10 --push
wait_version scoring "$TAG3" latest-
kubectl get isvc scoring-model -n scoring
for _ in $(seq 1 200); do
  curl -s -H "Host: $(host scoring)" "${GW}/version" | grep -oE '"version":"[^"]+"' || true
done | sort | uniq -c
gitops canary 0 --push # retour arrière (trafic seulement), puis promotion
sleep 90
gitops promote --push
wait_app scoring-prod
for _ in $(seq 1 20); do
  curl -s -H "Host: $(host scoring)" "${GW}/version" | grep -oE '"version":"[^"]+"' || true
done | sort | uniq -c

step "9. Démo complète : dérive -> ré-entraînement -> prod, sans intervention"
start_webhook 1
START=$SECONDS
uv run python -m pipelines.trigger_ct simulate --factor 2
uv run python -m pipelines.trigger_ct detect | tee /tmp/ct-detect.log
RUN_ID=$(sed -n 's/.*"run_id": "\([^"]*\)".*/\1/p' /tmp/ct-detect.log | tail -1)
wait_ct "$RUN_ID" released
echo "Cycle complet : $((SECONDS - START)) s"
git pull -q --rebase e2e main
git log --oneline -6
gitops status
test -s reports/rapport-m6.md || cp labs/m6-cicd/starter/rapport-m6.md reports/rapport-m6.md

step "Vérification automatique"
bash labs/m6-cicd/check.sh
