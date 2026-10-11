#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 8 (CI hebdomadaire) — sans intervention.
# Lancer depuis la racine du repo (branche solution/m8) : ./labs/m8-governance/e2e.sh
# Pré-requis : labs 0 à 7 rejoués sur le cluster (MLflow avec scoring-model@champion,
# image localhost:5001/scoring-api:1.0.0, InferenceService scoring-model dans le namespace
# scoring, géré par Argo CD depuis le dépôt Git de l'e2e du Lab 6, namespace gitops-e2e) ;
# outils uv, docker, kubectl, helm, curl.
# Image durcie (étape 1.3) : base dhi.io, accessible avec un compte Docker. Définir
# DHI_USERNAME et DHI_TOKEN (secrets CI) ; sans accès à dhi.io, l'étape est sautée (SKIP)
# et check.sh signalera les vérifications de l'image distroless.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source infra/versions.env
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS=ignore
GIT_ID=(-c user.name="CI Lab 8" -c user.email="ci@example.invalid")
CTX="kind-${KIND_CLUSTER_NAME}"
NS=scoring
SA=system:serviceaccount:${NS}:scoring-sa
GIT_PORT=19418 # port-forward du dépôt Git de l'e2e du Lab 6 (namespace gitops-e2e)
OVERLAY=k8s/overlays/prod/kustomization.yaml
k() { kubectl --context "$CTX" "$@"; }

log() { printf '\n=== %s\n' "$*"; }
skip() { printf 'SKIP (%s)\n' "$*"; }

wait_http() { # wait_http URL [essais]
  local url=$1 tries=${2:-60}
  for _ in $(seq 1 "$tries"); do
    if curl -s -o /dev/null -m 5 "$url"; then return 0; fi
    sleep 2
  done
  echo "Timeout en attendant ${url}" >&2
  return 1
}

log "Préalable — port-forwards, .env, venv"
[ -f .env ] || cp .env.example .env
set -a
# shellcheck disable=SC1091
source .env
set +a
make pf
wait_http "${MLFLOW_TRACKING_URI}/health"
uv sync
mkdir -p reports model-card governance k8s
uv run python -c "import fairlearn, shap; print(fairlearn.__version__, shap.__version__)"

log "Étape 1 — Trivy sur l'image du Lab 4, puis image distroless"
trivy() {
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    -v trivy-cache:/root/.cache/ "aquasec/trivy:${TRIVY_VERSION#v}" "$@"
}
docker pull localhost:5001/scoring-api:1.0.0
trivy image localhost:5001/scoring-api:1.0.0 > reports/trivy_full.txt
trivy image --severity HIGH,CRITICAL localhost:5001/scoring-api:1.0.0 \
  > reports/trivy_critical.txt
grep "^Total" reports/trivy_full.txt reports/trivy_critical.txt
if [ -n "${DHI_USERNAME:-}" ] && [ -n "${DHI_TOKEN:-}" ]; then
  echo "$DHI_TOKEN" | docker login dhi.io -u "$DHI_USERNAME" --password-stdin
fi
if docker pull -q dhi.io/python:3.12-debian13 > /dev/null 2>&1; then
  MODEL_VERSION=$(uv run python -c \
    "import json; print(json.load(open('packaging/model/metadata.json'))['model_version'])")
  docker build -f packaging/Dockerfile.distroless -t scoring-api:distroless \
    --build-arg VERSION=1.0.0 --build-arg MODEL_VERSION="$MODEL_VERSION" .
  docker run --rm --entrypoint python scoring-api:distroless --version
  if docker run --rm --entrypoint sh scoring-api:distroless -c true 2> /dev/null; then
    echo "Un shell est présent dans l'image distroless" >&2
    exit 1
  fi
  uv run python packaging/sample_request.py > /tmp/m8-request.json
  docker run -d --rm -p 8001:8000 --name m8-e2e-distroless scoring-api:distroless
  wait_http http://localhost:8001/health 30
  curl -fsS -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
    -d @/tmp/m8-request.json
  echo
  docker stop m8-e2e-distroless
  trivy image --severity HIGH,CRITICAL scoring-api:distroless > reports/trivy_distroless.txt
  grep "^Total" reports/trivy_critical.txt reports/trivy_distroless.txt
  docker tag scoring-api:distroless localhost:5001/scoring-api:1.0.0-distroless
  docker push localhost:5001/scoring-api:1.0.0-distroless
else
  skip "image distroless : dhi.io inaccessible (définir DHI_USERNAME et DHI_TOKEN)"
fi

log "Étape 2 — ModelScan : pickle vs ONNX vs skops"
uv run python packaging/scan-demo/make_pickles.py
rc=0
uv run modelscan -p packaging/scan-demo/ > reports/modelscan_pickle.txt || rc=$?
[ "$rc" -eq 1 ] # problème trouvé : le pickle piégé
rc=0
uv run modelscan -p packaging/model/model.onnx > reports/modelscan_onnx.txt || rc=$?
[ "$rc" -eq 3 ] # ONNX : aucun fichier pris en charge (« skipped »)
uv run modelscan -p packaging/scan-demo/ -r json -o reports/modelscan_pickle.json || true
grep -E "Total Issues|CRITICAL|Description" reports/modelscan_pickle.txt
grep -E "No issues|skipped" reports/modelscan_onnx.txt
rm -rf /tmp/m8-champion
uv run mlflow artifacts download -u models:/scoring-model@champion -d /tmp/m8-champion
uv run python -c "from skops.io import get_untrusted_types; \
print(get_untrusted_types(file='/tmp/m8-champion/model.skops'))"

log "Étape 3 — Fairlearn : disparités par attribut sensible"
uv run python -m governance.fairness_analysis | tee reports/fairness_baseline.txt

log "Étape 4 — Atténuations"
uv run python -m governance.fairness_mitigation | tee reports/fairness_mitigated.txt

log "Étape 5 — SHAP global + local"
uv run python -m governance.explain
uv run python -m governance.explain_local
ls reports/shap_waterfall_*.png

log "Étape 6 — Model Card"
skip "rédaction : Model Card du corrigé (model-card/model-card.md)"
if grep -q "TODO" model-card/model-card.md; then exit 1; fi
grep "^## " model-card/model-card.md

log "Étape 7 — RBAC minimal (ServiceAccount de l'ISVC de prod : par Git, comme l'énoncé)"
k apply -f k8s/rbac-scoring.yaml
# L'ISVC de prod est géré par Argo CD (Application scoring-prod, selfHeal) : un kubectl patch
# serait annulé. On pousse l'overlay prod sur le dépôt Git servi dans le cluster (e2e du Lab 6) :
# ce push sur main tient lieu de la fusion de la PR du lab (étape 9 de l'énoncé).
k get namespace gitops-e2e > /dev/null ||
  { echo "Rejouer d'abord l'e2e du Lab 6 (dépôt Git gitops-e2e lu par Argo CD)" >&2; exit 1; }
k -n gitops-e2e port-forward svc/git-server "${GIT_PORT}:9418" > /dev/null 2>&1 &
PF_GIT=$!
trap 'kill "$PF_GIT" 2> /dev/null || true' EXIT
sleep 3
git remote get-url e2e > /dev/null 2>&1 ||
  git remote add e2e "git://localhost:${GIT_PORT}/mlops-cours-2026.git"
git "${GIT_ID[@]}" pull -q --rebase --autostash e2e main
if ! k kustomize "$(dirname "$OVERLAY")" | grep -q "serviceAccountName: scoring-sa"; then
  cat >> "$OVERLAY" << 'EOF'
  # Lab 8 : identité minimale de l'API (ServiceAccount de k8s/rbac-scoring.yaml, appliqué
  # avant ce commit). En prod seulement : scoring-sa n'existe que dans le namespace scoring.
  # Dans Git, pas par kubectl patch : Argo CD (selfHeal) annulerait la modification.
  - target:
      kind: InferenceService
      name: scoring-model
    patch: |-
      - op: add
        path: /spec/predictor/serviceAccountName
        value: scoring-sa
EOF
fi
k kustomize "$(dirname "$OVERLAY")" | grep "serviceAccountName: scoring-sa"
uv run pytest -q tests/test_k8s.py
git add k8s/rbac-scoring.yaml "$OVERLAY"
git "${GIT_ID[@]}" commit -qm "feat(k8s): ServiceAccount minimal scoring-sa pour l'API de prod (Lab 8)" ||
  true
git push -q e2e HEAD:main
k annotate application scoring-prod -n argocd argocd.argoproj.io/refresh=normal --overwrite
k wait isvc/scoring-model -n "$NS" --timeout=300s \
  --for=jsonpath='{.spec.predictor.serviceAccountName}'=scoring-sa
k wait --for=condition=Ready "isvc/scoring-model" -n "$NS" --timeout=300s
[ "$(k auth can-i create pods --as="$SA" -n "$NS")" = "no" ]
[ "$(k auth can-i list secrets --as="$SA" -n "$NS")" = "no" ]
[ "$(k auth can-i get secret/s3-credentials --as="$SA" -n "$NS")" = "yes" ]
[ "$(k auth can-i get secret/s3-credentials --as="$SA" -n storage)" = "no" ]

log "Étape 8 — External Secrets Operator + ExternalSecret"
make external-secrets
k apply -f k8s/mlflow-creds-source.yaml
k wait --for=condition=Ready secretstore/course-secrets -n "$NS" --timeout=120s
k wait --for=condition=Ready externalsecret/s3-credentials -n "$NS" --timeout=120s
k get secretstore,externalsecret -n "$NS"
k delete secret s3-credentials -n "$NS"
for _ in $(seq 1 30); do
  k get secret s3-credentials -n "$NS" > /dev/null 2>&1 && break
  sleep 3
done
k get secret s3-credentials -n "$NS" # recréé par l'opérateur

log "Étape 9 — Rapport de conformité + commit"
skip "rédaction : rapport du corrigé (reports/compliance-ai-act.md)"
if grep -q "TODO" reports/compliance-ai-act.md; then exit 1; fi
git add governance model-card k8s packaging/Dockerfile.distroless \
  packaging/Dockerfile.distroless.dockerignore reports \
  ':(exclude)packaging/scan-demo/*.pkl'
git "${GIT_ID[@]}" commit -qm "feat(lab8): audit de gouvernance" || true
skip "branche lab/m8-governance + pull request : pas de push vers GitHub depuis la CI"

log "Vérification automatique"
bash labs/m8-governance/check.sh
