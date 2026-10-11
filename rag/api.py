# Fichier : rag/api.py
import time

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from config import COLLECTION, LLM_MODEL
from guardrails import safe_query
from rag import describe_source, vstore

app = FastAPI(title="RAG MLOps", version="1.0.0")

# Métriques LLM scrapables par Prometheus (Module 7)
REQUESTS = Counter("rag_requests", "Requêtes RAG", ["status"])  # ok | blocked | error
LATENCY = Histogram(
    "rag_request_duration_seconds",
    "Latence de bout en bout d'une requête RAG",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
TOKENS = Counter("rag_llm_tokens", "Tokens consommés par le LLM", ["direction"])


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    points = vstore.client.count(COLLECTION).count
    return {"status": "ok", "model": LLM_MODEL, "collection": COLLECTION, "points": points}


@app.post("/ask")
def ask(q: Question):
    start = time.perf_counter()
    try:
        result = safe_query(q.question)
    except Exception:
        REQUESTS.labels(status="error").inc()
        raise
    LATENCY.observe(time.perf_counter() - start)
    REQUESTS.labels(status="blocked" if result["blocked"] else "ok").inc()
    usage = getattr(result.get("message"), "usage_metadata", None) or {}
    TOKENS.labels(direction="input").inc(usage.get("input_tokens", 0))
    TOKENS.labels(direction="output").inc(usage.get("output_tokens", 0))
    return {
        "answer": result["result"],
        "blocked": result["blocked"],
        "sources": [describe_source(d) for d in result["source_documents"]],
        "latency_s": round(time.perf_counter() - start, 2),
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
