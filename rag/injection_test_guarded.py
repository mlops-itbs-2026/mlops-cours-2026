# Fichier : rag/injection_test_guarded.py
from guardrails import safe_query
from injection_test import attacks

queries = attacks + ["Que sont les 4 signaux d'or ?"]  # + 1 requête légitime

for query in queries:
    print(f"\n--- {query[:60]}... ---")
    result = safe_query(query)
    print(f"Response: {result['result'][:200]}")
