#!/usr/bin/env bash
# Rejeu automatique du Lab 9 (CI hebdomadaire) depuis la racine du repo assemblé (solution/m9).
# Le Lab 9 tourne sur le POSTE (Docker + Python), pas dans le cluster kind.
# Pour rester rapide en CI : petit modèle Ollama (qwen3.5:0.8b, ~1 Go) et golden set réduit.
set -euo pipefail
# shellcheck disable=SC1091
[ -f infra/versions.env ] && set -a && . infra/versions.env && set +a

OLLAMA_TAG="${OLLAMA_VERSION:-v0.35.1}"
OLLAMA_IMAGE="ollama/ollama:${OLLAMA_TAG#v}" # les tags Docker d'Ollama n'ont pas de « v »
QDRANT_IMAGE="qdrant/qdrant:${QDRANT_VERSION:-v1.19.2}"
E2E_MODEL="${E2E_MODEL:-qwen3.5:0.8b}"
E2E_RAGAS_LIMIT="${E2E_RAGAS_LIMIT:-3}"
WORK=$(mktemp -d)
API_PID="" PHX_PID=""

cleanup() {
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
  [ -n "$PHX_PID" ] && kill "$PHX_PID" 2>/dev/null || true
  docker rm -f e2e-qdrant e2e-ollama >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_http() { # wait_http URL secondes
  for _ in $(seq 1 "$2"); do curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "Timeout en attendant $1" >&2
  return 1
}

echo "== Étape 1 : LLM (Ollama CPU $OLLAMA_IMAGE, modèle $E2E_MODEL)"
echo "SKIP (Option B vLLM : GPU NVIDIA requis, indisponible en CI)"
docker rm -f e2e-ollama >/dev/null 2>&1 || true
docker run -d --name e2e-ollama -p 11434:11434 "$OLLAMA_IMAGE" >/dev/null
wait_http http://localhost:11434/api/version 120
docker exec e2e-ollama ollama pull "$E2E_MODEL"
export LLM_BACKEND=ollama LLM_MODEL="$E2E_MODEL"

echo "== Étape 2 : Qdrant ($QDRANT_IMAGE)"
docker rm -f e2e-qdrant >/dev/null 2>&1 || true
docker run -d --name e2e-qdrant -p 6333:6333 -p 6334:6334 "$QDRANT_IMAGE" >/dev/null
wait_http http://localhost:6333/ 60

# Le RAG est un projet uv isolé (rag/pyproject.toml + rag/uv.lock) : tout se lance depuis rag/
cd rag
mkdir -p reports

echo "== Étape 3.2 : dépendances Python (projet isolé rag/, venv rag/.venv)"
uv sync

echo "== Étape 7 (avant tout appel) : serveur Phoenix"
PHOENIX_WORKING_DIR="$WORK/phoenix" uv run phoenix serve >"$WORK/phoenix.log" 2>&1 &
PHX_PID=$!
wait_http http://localhost:6006/ 120

echo "== Étape 3 : ingestion (pas de PDF du cours en CI : énoncés labs/*/README.md)"
uv run python ingest.py --with-labs

echo "== Étape 4 : chaîne RAG (tracée dans Phoenix)"
PHOENIX_TRACING=1 uv run python rag.py

echo "== Étape 6 : Ragas sur $E2E_RAGAS_LIMIT questions (juge = $E2E_MODEL)"
RAGAS_LIMIT="$E2E_RAGAS_LIMIT" uv run python evaluate_rag.py 2>&1 | tee reports/ragas_run.log

echo "== Étape 8 : prompt injection avant / après guardrail"
uv run python injection_test.py 2>&1 | tee reports/injection_before.log
uv run python injection_test_guarded.py 2>&1 | tee reports/injection_after.log

echo "== Étape 7.2 : API RAG (tracée)"
PHOENIX_TRACING=1 uv run uvicorn api:app --port 8090 >"$WORK/api.log" 2>&1 &
API_PID=$!
wait_http http://localhost:8090/health 180
for q in "Que fait PagedAttention dans vLLM ?" "Qu'est-ce que le burn rate d'un SLO ?"; do
  curl -sf -m 300 -X POST http://localhost:8090/ask -H 'Content-Type: application/json' \
    -d "{\"question\": \"$q\"}"
  echo
done
sleep 5 # laisser le BatchSpanProcessor exporter les dernières traces

echo "== Étape 7.1 : chiffres Phoenix (tokens, coût, latence)"
uv run python phoenix_stats.py

echo "== Étape 9 : RAPPORT.md"
echo "SKIP (rédaction et capture d'écran Phoenix : travail de l'étudiant, trame de la solution)"
cd ..

echo "== Vérification"
# Golden set réduit et petit modèle juge : seuils assouplis (on teste la plomberie, pas la qualité)
M9_MIN_RAGAS_ROWS="$E2E_RAGAS_LIMIT" M9_MIN_FAITHFULNESS=-1 ./labs/m9-llmops/check.sh
