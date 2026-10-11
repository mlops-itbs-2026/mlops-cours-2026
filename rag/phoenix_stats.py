# Fichier : rag/phoenix_stats.py
# Chiffres du rapport (tokens, coût, latence) calculés à partir des traces Phoenix
import json
import os

from phoenix.client import Client

from config import REPORTS_DIR

PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "rag-mlops")
# Tarif de référence pour l'équivalent API managée (USD par million de tokens).
# GPT-4o-mini : 0,15 $ en entrée, 0,60 $ en sortie — à vérifier sur la grille du fournisseur.
PRICE_IN, PRICE_OUT = 0.15, 0.60

client = Client(base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
spans = client.spans.get_spans_dataframe(project_identifier=PROJECT, limit=10_000)

llm = spans[spans["span_kind"] == "LLM"]


def total(column: str) -> int:
    return int(llm[column].fillna(0).sum()) if column in llm else 0


tokens_in = total("attributes.llm.token_count.prompt")
tokens_out = total("attributes.llm.token_count.completion")

# 1 requête RAG = 1 span CHAIN LangChain « de premier niveau » : sans parent, ou dont le parent
# n'est pas un span LangChain (dans l'API, FastAPI crée lui-même un span HTTP parent)
langchain_ids = set(spans.index[spans["span_kind"] != "UNKNOWN"])
roots = spans[(spans["span_kind"] == "CHAIN") & ~spans["parent_id"].isin(langchain_ids)]
latency = (roots["end_time"] - roots["start_time"]).dt.total_seconds()

stats = {
    "rag_requests": len(roots),
    "llm_calls": len(llm),
    "tokens_in": tokens_in,
    "tokens_out": tokens_out,
    "cost_equiv_usd": round((tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6, 4),
    "latency_p50_s": round(float(latency.quantile(0.50)), 2),
    "latency_p95_s": round(float(latency.quantile(0.95)), 2),
}
print(json.dumps(stats, indent=2))
REPORTS_DIR.mkdir(exist_ok=True)
(REPORTS_DIR / "phoenix_stats.json").write_text(json.dumps(stats, indent=2))
