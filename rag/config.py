# Fichier : rag/config.py
"""Configuration commune du RAG : UN seul endroit pour Qdrant, les embeddings et le LLM.

L'ingestion, la chaîne, l'évaluation et l'API DOIVENT utiliser le même modèle
d'embeddings et la même collection. Tout se règle par variables d'environnement.
"""

import os
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
REPO_ROOT = RAG_DIR.parent
REPORTS_DIR = RAG_DIR / "reports"

# --- Qdrant -------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH")  # mode local sans serveur (tests) si défini
COLLECTION = os.getenv("QDRANT_COLLECTION", "mlops_course")

# --- Embeddings : bge-small-en-v1.5 (384 dim.), ONNX via fastembed, CPU, sans PyTorch
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- LLM ----------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # ollama | vllm | fake
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)  # juge Ragas (par défaut : le même)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

NO_ANSWER = "Je n'ai pas cette information dans les documents du cours."


def qdrant_location() -> dict:
    """Arguments de connexion Qdrant pour langchain-qdrant (serveur ou dossier local)."""
    if QDRANT_PATH:
        return {"path": QDRANT_PATH}
    return {"url": QDRANT_URL}


def get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings

    # Téléchargé depuis Hugging Face au premier lancement (~70 Mo), puis en cache
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def get_llm(temperature: float = 0.1):
    """LLM de la chaîne RAG (interface LangChain)."""
    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama

        # reasoning=False : désactive le mode « thinking » (Qwen 3.5…), inutile ici et lent
        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            reasoning=False,
        )
    if LLM_BACKEND == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=VLLM_BASE_URL,
            model=LLM_MODEL,  # ex. Qwen/Qwen3.5-4B (nom servi par vllm serve)
            api_key=os.getenv("VLLM_API_KEY", "not-needed"),
            temperature=temperature,
        )
    if LLM_BACKEND == "fake":  # tests automatiques sans LLM (CI, check rapide)
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=[NO_ANSWER])
    raise ValueError(f"LLM_BACKEND inconnu : {LLM_BACKEND}")


def get_judge_llm():
    """LLM juge pour Ragas 0.4 (API moderne : llm_factory + client compatible OpenAI).

    Ollama et vLLM exposent tous les deux une API compatible OpenAI (/v1).
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    if LLM_BACKEND == "vllm":
        client = AsyncOpenAI(
            base_url=VLLM_BASE_URL, api_key=os.getenv("VLLM_API_KEY", "not-needed"), timeout=600
        )
        return llm_factory(JUDGE_MODEL, provider="openai", client=client, max_tokens=2048)
    client = AsyncOpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama", timeout=600)
    # reasoning_effort="none" : équivalent OpenAI de think=false côté Ollama
    return llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=2048,
        reasoning_effort="none",
    )
