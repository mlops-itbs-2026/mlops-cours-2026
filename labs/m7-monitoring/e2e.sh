#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 7 (CI du cours) — Monitoring & observabilité ML
# Pré-requis : labs 0 à 6 rejoués sur ce runner, dont l'e2e du Lab 6 (dépôt Git servi
# dans le cluster, namespace gitops-e2e, remote « e2e » ; Applications Argo CD scoring-prod
# et scoring-staging), registry localhost:5001, MLflow + KFP, port-forwards (make pf).
# Ce qu'un runner ne peut pas faire est remplacé, et signalé par SKIP :
#   - GitHub Actions -> build local (pipelines.gitops.local_ci, comme l'e2e du Lab 6) ;
#   - Slack / webhook.site -> URL factice (la route est chargée, aucun message envoyé).
# Options : KFP_PORT (8080), E2E_DRIFT_SECONDS (trafic dérivé avant l'analyse, défaut 300).
# Lance ensuite labs/m7-monitoring/check.sh ; code retour = celui du check.
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
NS=scoring
MON=monitoring
KFP_PORT=${KFP_PORT:-8080}
DRIFT_SECONDS=${E2E_DRIFT_SECONDS:-300}
GIT_PORT=19418
ISVC_HOST=scoring-model-predictor.${NS}.example.com
PROM="/api/v1/namespaces/${MON}/services/http:kube-prom-stack-kube-prome-prometheus:9090/proxy"
export KFP_HOST="http://localhost:${KFP_PORT}" GATEWAY="$GW" ISVC_HOST
export GITOPS_REMOTE=e2e CT_LOCAL_CI=1 CT_COOLDOWN_SECONDS=0 MLFLOW_DISABLE_TELEMETRY=true
PIDS=()
TRAFFIC_PID=""
step() { echo; echo "=== $* ==="; }
skip() { echo "SKIP ($*)"; }
cleanup() {
  [ -z "$TRAFFIC_PID" ] || kill "$TRAFFIC_PID" >/dev/null 2>&1 || true
  [ "${#PIDS[@]}" -eq 0 ] || kill "${PIDS[@]}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
gitops() { uv run python -m pipelines.gitops "$@"; }
version_of() { curl -fsS --max-time 30 -H "Host: ${2:-}${ISVC_HOST}" "${GW}/version"; }
wait_version() { # wait_version <tag> [route] : jusqu'à 10 min
  for _ in $(seq 1 60); do
    version_of "$1" "${2:-}" 2>/dev/null | grep -q "\"version\":\"$1\"" && return 0
    sleep 10
  done
  echo "Délai dépassé : la prod ne sert pas $1 ${2:+(route $2)}" >&2
  return 1
}
local_ci() { # build + push + overlay staging (jobs de ci.yml sans GitHub) ; affiche le tag
  local sha mv
  sha=$(git rev-parse HEAD)
  mv=$(jq -r .model_version packaging/model/metadata.json)
  uv run python -c "from pathlib import Path; from pipelines import gitops; \
print(gitops.local_ci(Path('.'), '${sha}', '${mv}', do_push=True))" | tail -1
}
prom_count() { # prom_count <PromQL> : nombre de séries
  kubectl get --raw "${PROM}/api/v1/query?query=$(jq -rn --arg q "$1" '$q|@uri')" |
    jq '.data.result | length'
}
wait_prom() { # wait_prom <PromQL> [min] : jusqu'à 5 min
  for _ in $(seq 1 30); do
    [ "$(prom_count "$1")" -ge "${2:-1}" ] && return 0
    sleep 10
  done
  echo "Délai dépassé : ${1}" >&2
  return 1
}
traffic() { # traffic <args…> : trafic en arrière-plan (un seul à la fois)
  [ -z "$TRAFFIC_PID" ] || kill "$TRAFFIC_PID" >/dev/null 2>&1 || true
  uv run python monitoring/traffic.py --url "$GW" --host-header "$ISVC_HOST" "$@" \
    >/tmp/m7-traffic.log 2>&1 &
  TRAFFIC_PID=$!
}
run_drift_job() { # run_drift_job <nom>
  kubectl delete job "$1" -n "$MON" --ignore-not-found
  kubectl create job --from=cronjob/scoring-drift "$1" -n "$MON"
  kubectl wait --for=condition=complete "job/$1" -n "$MON" --timeout=600s
  kubectl logs -n "$MON" "job/$1"
}

step "0. Environnement Python, dépôt GitOps de l'e2e du Lab 6"
uv sync
git config user.name >/dev/null || git config user.name "CI du cours"
git config user.email >/dev/null || git config user.email "ci@mlops-cours.invalid"
kubectl get namespace gitops-e2e >/dev/null ||
  { echo "Rejouer d'abord l'e2e du Lab 6 (dépôt Git gitops-e2e)" >&2; exit 1; }
kubectl -n gitops-e2e port-forward svc/git-server "${GIT_PORT}:9418" >/dev/null 2>&1 &
PIDS+=($!)
sleep 3
git remote get-url e2e >/dev/null 2>&1 ||
  git remote add e2e "git://localhost:${GIT_PORT}/mlops-cours-2026.git"
git pull -q --rebase e2e main

step "1. API instrumentée : tests et essai local"
LOG_PREDICTIONS=false uv run pytest -q tests/test_monitoring.py
uv run python -m uvicorn app.serve:app --app-dir packaging --port 18700 >/tmp/m7-api.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do curl -fsS localhost:18700/health >/dev/null 2>&1 && break; sleep 1; done
uv run python packaging/sample_request.py -n 20 >/tmp/m7-request.json
for _ in $(seq 1 20); do
  curl -fsS -X POST localhost:18700/predict -H "Content-Type: application/json" \
    -d @/tmp/m7-request.json >/dev/null
done
curl -fsS localhost:18700/metrics | grep -E '^ml_predictions_total'
grep -c '"event":"prediction"' /tmp/m7-api.log

step "2. Déploiement par GitOps : CI (locale) -> staging -> canary 10 % -> promotion"
skip "GitHub Actions : image construite localement (comme l'e2e du Lab 6)"
git add packaging monitoring tests reports labs/m7-monitoring 2>/dev/null || true
git commit -qm "feat(api): métriques Prometheus et journal des prédictions (Lab 7)" || true
git push -q e2e HEAD:main
TAG=$(local_ci)
git pull -q --rebase e2e main
gitops deploy --env prod --canary 10 --push
wait_version "$TAG" latest-
curl -fsS -H "Host: latest-${ISVC_HOST}" "${GW}/metrics" | grep '^ml_model_info'
gitops promote --push
wait_version "$TAG"

step "3. Stack d'observabilité"
make monitoring
kubectl get pods -n "$MON"
./infra/monitoring/port-forward.sh

step "4. PodMonitor"
kubectl apply -f monitoring/scoring-podmonitor.yaml
traffic --duration 0 # trafic normal, jusqu'à l'étape 9
wait_prom "up{namespace=\"${NS}\",isvc=\"scoring-model\"} == 1"
wait_prom "ml_predictions_total{namespace=\"${NS}\"}"

step "5. Dashboard versionné"
kubectl apply -k monitoring/dashboards
kubectl get configmap -n "$MON" -l grafana_dashboard=1

step "6. Job de drift (Evidently) : image, configuration, CronJob"
docker build -f monitoring/evidently-job/Dockerfile -t "${REG}/scoring-drift:1.0.0" .
docker push "${REG}/scoring-drift:1.0.0"
kubectl create secret generic s3-credentials -n "$MON" \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
uv run dvc push data/processed/train.csv
REFERENCE_URI=$(uv run python -c "import dvc.api; print(dvc.api.get_url('data/processed/train.csv'))")
echo "Référence : ${REFERENCE_URI}"
kubectl create configmap scoring-drift-reference -n "$MON" \
  --from-literal=REFERENCE_URI="$REFERENCE_URI" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f monitoring/evidently-cronjob.yaml
sleep 60 # ≥ 500 demandes journalisées (20 req/s)
run_drift_job manual-run-1
wait_prom 'ml_drift_score{model="scoring-model"}'

step "7. Loki + Alloy"
# shellcheck disable=SC2016  # les accents graves sont la syntaxe LogQL, pas du shell
LOGQL='{namespace="scoring", container="kserve-container"} |= `"event":"prediction"`'
curl -fsS -G localhost:3100/loki/api/v1/query_range --data-urlencode limit=1 \
  --data-urlencode "query=${LOGQL}" | jq -e '.data.result | length > 0'

step "8. Alertes et Alertmanager"
kubectl apply -f monitoring/scoring-alerts.yaml
skip "Slack / webhook.site : URL factice, la route est chargée mais aucun message n'est envoyé"
kubectl create secret generic slack-webhook -n "$MON" \
  --from-literal=url="http://slack-e2e.invalid/services/T000/B000/XXXX" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f monitoring/alertmanager-config.yaml
for _ in $(seq 1 30); do
  curl -fsS localhost:9093/api/v2/status | grep -q 'monitoring/scoring-slack/slack-ml' && break
  sleep 10
done
curl -fsS localhost:9093/api/v2/status | grep -o 'monitoring/scoring-slack/slack-ml[a-z-]*' |
  sort -u
kubectl get --raw "${PROM}/api/v1/rules?type=alert" |
  jq -e '[.data.groups[] | select(.name == "scoring.alerts") | .rules[]] | length == 6'

step "9. Démo : drift -> alerte -> ré-entraînement"
traffic --mode drift --duration 0 # sans --drift : variables numériques x 1,5 (dataset.yaml)
sleep "$DRIFT_SECONDS"
run_drift_job manual-run-2
wait_prom 'ml_drift_detected{model="scoring-model"} == 1'
kubectl get --raw "${PROM}/api/v1/rules?type=alert" |
  jq -r '.data.groups[] | select(.name == "scoring.alerts") | .rules[] | "\(.name) \(.state)"'
if ! curl -fsS "${KFP_HOST}/apis/v2beta1/healthz" >/dev/null 2>&1; then
  kubectl -n kubeflow port-forward svc/ml-pipeline-ui "${KFP_PORT}:80" >/dev/null 2>&1 &
  PIDS+=($!)
  sleep 3
fi
grep -q '^CT_WEBHOOK_TOKEN=' .env 2>/dev/null ||
  echo "CT_WEBHOOK_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(16))')" >>.env
CT_WEBHOOK_TOKEN=$(sed -n 's/^CT_WEBHOOK_TOKEN=//p' .env | tail -1)
export CT_WEBHOOK_TOKEN
# Pas de mise en production automatique ici : l'étape 10 crée son propre canary.
CT_AUTO_RELEASE=0 uv run python -m pipelines.trigger_ct serve >/tmp/m7-ct-webhook.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do curl -fsS localhost:8095/health >/dev/null 2>&1 && break; sleep 2; done
REFERENCE_URI="$REFERENCE_URI" LOKI_URL=http://localhost:3100 PUSHGATEWAY_URL="" REPORTS_URI="" \
  CT_WEBHOOK_URL=http://localhost:8095/drift \
  uv run python monitoring/evidently-job/calc_drift.py | tee /tmp/m7-calc-drift.log
grep -q 'Webhook CT .* HTTP 20[02]' /tmp/m7-calc-drift.log

step "10. Canary analysé par les métriques"
traffic --duration 0 # retour au trafic normal
git commit -q --allow-empty -m "chore: nouvelle version pour le canary analysé (Lab 7)"
git push -q e2e HEAD:main
TAG2=$(local_ci)
git pull -q --rebase e2e main
gitops deploy --env prod --canary 10 --push
wait_version "$TAG2" latest-
sleep 180
uv run python monitoring/canary_analysis.py --apply --push # CONTINUE -> 50 %
sleep 150
uv run python monitoring/canary_analysis.py --apply --push # PROMOTE -> 100 %
wait_version "$TAG2"
# Canary défectueux : l'image du Lab 5 n'expose pas /metrics -> ROLLBACK (code 1)
gitops deploy --env prod --canary 10 --image "${REG}/scoring-api" --tag 2.0.0 --push
wait_version 2.0.0 latest-
sleep 120
rc=0
uv run python monitoring/canary_analysis.py --apply --push || rc=$?
[ "$rc" -eq 1 ] || { echo "ROLLBACK attendu (code 1), obtenu ${rc}" >&2; exit 1; }
git pull -q --rebase e2e main
gitops deploy --env prod --canary 100 --push # retour à la bonne version (staging), à 100 %
wait_version "$TAG2"
git log --oneline -6
kubectl get isvc scoring-model -n "$NS"
test -s reports/rapport-m7.md || cp labs/m7-monitoring/starter/rapport-m7.md reports/rapport-m7.md

step "Vérification automatique"
bash labs/m7-monitoring/check.sh
