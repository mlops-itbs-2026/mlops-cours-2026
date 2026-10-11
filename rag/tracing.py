# Fichier : rag/tracing.py
# Importé par rag.py quand PHOENIX_TRACING=1
import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

# Le serveur Phoenix (« phoenix serve ») reçoit les traces OTLP/HTTP sur /v1/traces
PHOENIX_URL = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

tracer_provider = register(
    project_name=os.getenv("PHOENIX_PROJECT_NAME", "rag-mlops"),
    endpoint=f"{PHOENIX_URL}/v1/traces",
    batch=True,  # export asynchrone, vidé automatiquement à la fin du script
    verbose=False,
)

# Auto-instrumentation LangChain : retriever (Qdrant), prompt, LLM et chaîne complète
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Maintenant chaque requête RAG est tracée
