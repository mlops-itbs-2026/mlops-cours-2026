# Fichier : rag/guardrails.py
import re
import unicodedata

from rag import qa

# Patterns de prompt injection courants (anglais + français, comparés sans accents)
INJECTION_PATTERNS = [
    r"ignore.*(previous|precedent|above|instruction)",
    r"oublie.*(instruction|regle|consigne)",
    r"system\s*override",
    r"(reveal|revele|affiche|montre).*(prompt|instruction)",
    r"forget.*(instructions|rules)",
    r"disregard.*(previous|above)",
    r"you are now",
    r"tu es (maintenant|desormais)",
    r"pretend to be",
    r"jailbreak",
    r"\bexecute\b.*(code|python|print|commande|shell)",
]

# Guardrail de sortie : ce qui ne doit JAMAIS sortir du RAG
OUTPUT_PATTERNS = {
    # phrase présente uniquement dans le prompt système (rag.py) : si elle sort, il y a fuite
    "System prompt leakage": r"entre les balises <contexte>",
    # secrets : clés d'accès, mots de passe affectés, clés d'API (LLM02, divulgation)
    "Secret in answer": (
        r"(secret_access_key|password|mot de passe)\s*[=:]\s*\S+|\bsk-[a-z0-9]{16,}"
    ),
}


def normalize(text: str) -> str:
    """Minuscules sans accents : « Précédentes » et « precedentes » sont traités pareil."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def check_injection(query: str) -> tuple[bool, str]:
    """Retourne (is_suspicious, reason)."""
    q = normalize(query)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True, f"Pattern detected: {pattern}"
    # Trop long ?
    if len(query) > 1000:
        return True, "Query > 1000 chars (possible stuffing)"
    # Trop de newlines ?
    if query.count("\n") > 5:
        return True, "Too many newlines (possible structure breaking)"
    return False, ""


def check_output(answer: str) -> tuple[bool, str]:
    """Retourne (is_unsafe, reason) pour la réponse du LLM."""
    a = normalize(answer)
    for reason, pattern in OUTPUT_PATTERNS.items():
        if re.search(pattern, a, re.IGNORECASE):
            return True, reason
    return False, ""


# Wrapper safe : guardrail en entrée, appel RAG, guardrail en sortie
def safe_query(query: str) -> dict:
    suspicious, reason = check_injection(query)
    if suspicious:
        return {
            "result": f"[BLOCKED] Query rejected: {reason}",
            "source_documents": [],
            "blocked": True,
        }
    result = qa.invoke({"query": query})
    unsafe, reason = check_output(result["result"])
    if unsafe:
        return {
            **result,
            "result": f"[BLOCKED] Answer rejected: {reason}",
            "source_documents": [],
            "blocked": True,
        }
    return {**result, "blocked": False}
