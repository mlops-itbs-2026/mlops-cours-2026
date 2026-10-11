# Fichier : rag/evaluate_rag.py
import asyncio
import json
import math
import os
import sys

import pandas as pd
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config import JUDGE_MODEL, RAG_DIR, REPORTS_DIR, get_embeddings, get_judge_llm
from rag import qa


class RagasEmbeddings(BaseRagasEmbedding):
    """Adaptateur : expose à Ragas les MÊMES embeddings que le retriever (fastembed)."""

    def __init__(self, lc_embeddings):
        super().__init__()
        self.lc = lc_embeddings

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.lc.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.lc.embed_query(text)


# Charger le golden set (RAGAS_LIMIT=5 pour un premier run rapide)
golden = json.loads((RAG_DIR / "golden_set.json").read_text(encoding="utf-8"))
limit = int(os.getenv("RAGAS_LIMIT", "0"))
if limit:
    golden = golden[:limit]

# Générer les réponses + contextes avec NOTRE chaîne RAG
records = []
for item in golden:
    print(f"Evaluating: {item['question'][:50]}...")
    result = qa.invoke({"query": item["question"]})
    records.append(
        {
            "user_input": item["question"],
            "response": result["result"],
            "retrieved_contexts": [d.page_content for d in result["source_documents"]],
            "reference": item["ground_truth"],
        }
    )

# Ragas a besoin d'un LLM pour juger (LLM-as-a-judge) : on réutilise notre LLM, à température 0
judge = get_judge_llm()
metrics = {
    "faithfulness": Faithfulness(llm=judge),
    # strictness=1 : 1 appel LLM au lieu de 3 (à température 0, les 3 seraient identiques)
    "answer_relevancy": AnswerRelevancy(
        llm=judge, embeddings=RagasEmbeddings(get_embeddings()), strictness=1
    ),
    "context_precision": ContextPrecision(llm=judge),
    "context_recall": ContextRecall(llm=judge),
}
# Chaque métrique ne reçoit que les champs dont elle a besoin
FIELDS = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "retrieved_contexts", "reference"),
}


async def score_all() -> None:
    # Une requête à la fois : un Ollama sur CPU ne sait pas en traiter plusieurs en parallèle
    for i, rec in enumerate(records, start=1):
        for name, metric in metrics.items():
            try:
                res = await metric.ascore(**{k: rec[k] for k in FIELDS[name]})
                rec[name] = res.value
            except Exception as e:  # sortie JSON invalide du juge, timeout…
                print(f"  [{i}] {name}: échec ({type(e).__name__}: {str(e)[:80]})")
                rec[name] = math.nan
        print(f"Scored {i}/{len(records)}: " + ", ".join(f"{m}={rec[m]:.2f}" for m in metrics))


print(f"\nJuge : {JUDGE_MODEL} — {len(records)} questions × {len(metrics)} métriques")
asyncio.run(score_all())

df = pd.DataFrame(records)
summary = {m: round(float(df[m].mean()), 3) for m in metrics}  # moyenne hors NaN

print("\n=== RESULTATS RAGAS ===")
print(summary)

# Sauver en CSV pour le rapport (+ résumé JSON pour la CI)
REPORTS_DIR.mkdir(exist_ok=True)
df.to_csv(REPORTS_DIR / "ragas_results.csv", index=False)
(REPORTS_DIR / "ragas_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nDetails sauves dans {REPORTS_DIR / 'ragas_results.csv'}")

# Gate optionnelle (CI) : RAGAS_MIN_FAITHFULNESS=0.7 -> code retour 1 si en dessous
threshold = os.getenv("RAGAS_MIN_FAITHFULNESS")
if threshold and not summary["faithfulness"] >= float(threshold):
    print(f"ECHEC : faithfulness {summary['faithfulness']} < {threshold}")
    sys.exit(1)
