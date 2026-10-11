#!/usr/bin/env bash
# =============================================================================
# Rejeu automatique du Lab 1 (CI hebdomadaire) — sans intervention.
# Lancer depuis la racine du repo (branche solution/m1) : ./labs/m1-versioning/e2e.sh
# Pré-requis : le Lab 0 a été rejoué (cluster, SeaweedFS, MLflow) ; uv, git, curl.
# Les fichiers du lab (src/, params.yaml, dvc.yaml, features/…) sont déjà dans le repo.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
export DVC_NO_ANALYTICS=1 MLFLOW_DISABLE_AGENT_HINT=1
GIT_ID=(-c user.name="CI Lab 1" -c user.email="ci@example.invalid")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

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
wait_http "${AWS_ENDPOINT_URL}"
uv sync
# shellcheck disable=SC1091
source .venv/bin/activate # `python` des commandes de dvc.yaml = le venv du cours

log "Étape 1 — contrat de données"
python -c "from src.dataset_config import load_config; print(load_config())" | cut -c1-200
python -m src.download
python - <<'EOF'
import pandas as pd

from src.dataset_config import load_config

cfg = load_config()
df = pd.read_csv(cfg.raw_path)
print(df.shape, df[cfg.target].value_counts(normalize=True).round(2).to_dict())
print(cfg.encode_target(df[cfg.target]).groupby(cfg.sensitive_frame(df).iloc[:, 0]).mean())
EOF

log "Étape 2 — DVC : stage download + remote S3"
[ -d .dvc ] || dvc init
dvc remote add -d -f storage s3://dvc-store/scoring
dvc remote modify storage endpointurl "${AWS_ENDPOINT_URL}"
dvc remote modify --local storage access_key_id "${AWS_ACCESS_KEY_ID}"
dvc remote modify --local storage secret_access_key "${AWS_SECRET_ACCESS_KEY}"
skip "dvc stage add : dvc.yaml complet déjà fourni par la solution"
dvc repro download
dvc push
git add .dvc/config .dvc/.gitignore .dvcignore dvc.yaml dvc.lock data/raw/.gitignore
git "${GIT_ID[@]}" commit -qm "data: stage download + remote S3" || true

log "Étape 3 — serveur MLflow du cours"
[ "$(curl -fsS "${MLFLOW_TRACKING_URI}/health")" = "OK" ]
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  python -c "import mlflow; print([e.name for e in mlflow.search_experiments()])"

log "Étape 4 — préparation + entraînement instrumenté"
python -m src.prepare
python -m src.train
python -m src.train --model random_forest --n_estimators 50 --max_depth 5

log "Étape 5 — grid search (22 runs)"
python -m src.grid_search

log "Étape 6 — comparaison des runs"
skip "UI MLflow : pas de navigateur en CI"
python - <<'EOF'
import mlflow

client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("scoring")
for r in client.search_runs([exp.experiment_id], order_by=["metrics.roc_auc DESC"],
                            max_results=5):
    print(r.info.run_name, round(r.data.metrics["roc_auc"], 4))
EOF

log "Étape 7 — Model Registry : alias et gate"
python -m src.promote
python -m src.promote --filter "params.model = 'dummy'"
python - <<'EOF'
import mlflow

from src.dataset_config import load_config
from src.train import TEST_CSV

# L'alias suffit : ni run, ni numéro de version, ni code d'entraînement
uri = "models:/scoring-model@champion"
model = mlflow.sklearn.load_model(uri)
threshold = mlflow.models.get_model_info(uri).metadata["decision_threshold"]
X_test, _ = load_config().read_xy(TEST_CSV)
proba = model.predict_proba(X_test.head(5))[:, 1]
print(f"seuil de décision : {threshold:.3f}")
print("P(bad) :", proba.round(2))
print("refus  :", (proba >= threshold).astype(int))
EOF

log "Étape 8 — pipeline DVC complet"
dvc repro
dvc metrics show
python -m src.promote --filter "tags.launched_by = 'dvc'"
git add dvc.yaml dvc.lock params.yaml src/ data/processed/.gitignore models/.gitignore \
  reports/metrics.json
git "${GIT_ID[@]}" commit -qm "feat: pipeline DVC complet" || true
sed -i.bak 's/^  class_weight: balanced/  class_weight: none/' params.yaml && rm -f params.yaml.bak
dvc repro
dvc params diff
dvc metrics diff
git checkout params.yaml
dvc repro | tee "$TMP/runcache.log"
grep -q "is cached" "$TMP/runcache.log" # restauré depuis le run-cache
[ -z "$(git status --porcelain dvc.lock params.yaml reports/)" ] # retour à l'état commité
dvc push

log "Étape 9 — feature store Feast"
(cd features && feast apply && feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)")
python - <<'EOF'
import pandas as pd
import yaml
from feast import FeatureStore

from src.dataset_config import load_config

cfg = load_config()
id_col = yaml.safe_load(open("params.yaml"))["prepare"]["id_column"]
store = FeatureStore(repo_path="features")
train = pd.read_csv("data/processed/train.csv")
entity_df = train[[id_col, cfg.target]].assign(event_timestamp=pd.Timestamp.now(tz="UTC"))
df = store.get_historical_features(
    entity_df=entity_df, features=store.get_feature_service("scoring_v1")
).to_df()
assert len(df) == len(train) and df[cfg.features].notna().all().all()
online = store.get_online_features(
    features=store.get_feature_service("scoring_v1"),
    entity_rows=[{id_col: int(train[id_col].iloc[0])}],
).to_dict()
print({c: online[c][0] for c in cfg.features[:4]})
EOF
git add features/feature_store.yaml features/definitions.py
git "${GIT_ID[@]}" commit -qm "feat: feature store Feast" || true

log "Étape 10 — reproductibilité depuis un clone vierge"
git add NOTES.md # instructions de repro (étape 10.1), fournies par la solution
git "${GIT_ID[@]}" commit -qm "docs: instructions de reproduction (Lab 1)" || true
skip "branche lab/m1-versioning + pull request : pas de push depuis la CI (clone du repo local)"
skip "étape 10.4 (autre poste) : un seul cluster en CI ; dvc repro --force rejoue tout ci-dessous"
git clone -q "$REPO" "$TMP/repro-test"
(
  cd "$TMP/repro-test"
  dvc remote modify --local storage access_key_id "${AWS_ACCESS_KEY_ID}"
  dvc remote modify --local storage secret_access_key "${AWS_SECRET_ACCESS_KEY}"
  dvc pull
  dvc repro | tee "$TMP/repro.log"
  if grep -q "Running stage" "$TMP/repro.log"; then exit 1; fi # tout doit être « skipping »
  dvc metrics show
  dvc repro --force
  dvc metrics diff --all
  [ -z "$(dvc metrics diff)" ] # mêmes métriques après ré-exécution complète
)

log "Vérification automatique"
bash labs/m1-versioning/check.sh
