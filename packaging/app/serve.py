# Fichier : packaging/app/serve.py
"""API de scoring : sert le modèle ONNX exporté à l'étape 1.

Le schéma d'entrée est construit à partir de dataset.yaml : aucune colonne n'est écrite
ici, changer de dataset ne demande aucune modification de ce fichier.

Lab 7 : l'API est instrumentée. GET /metrics expose des métriques Prometheus (trafic,
latence, erreurs, scores, décisions, décisions par groupe sensible) et chaque prédiction
est journalisée en JSON sur la sortie standard (collectée par Alloy -> Loki, relue par le
job de drift).

Lancement local (depuis la racine du repo) :
    uv run python -m uvicorn app.serve:app --app-dir packaging --port 8000
"""

import bisect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    GC_COLLECTOR,
    PLATFORM_COLLECTOR,
    PROCESS_COLLECTOR,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
    disable_created_metrics,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parents[2]  # racine du repo (/app dans l'image)
sys.path.append(str(ROOT))  # rend « src » importable quel que soit le lanceur
from src.dataset_config import load_config  # noqa: E402

# Version de l'image : ENV APP_VERSION posée au build (Lab 5) ; 1.0.0 pour l'image du Lab 4
API_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "packaging" / "model" / "model.onnx"))

cfg = load_config(os.getenv("DATASET_CONFIG", ROOT / "dataset.yaml"))
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
meta_path = MODEL_PATH.with_name("metadata.json")
metadata: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
# Seuil : variable d'environnement > metadata.json (seuil du run d'entraînement) > coût métier
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", metadata.get("threshold", cfg.cost_threshold())))
MODEL_VERSION = str(metadata.get("model_version", "unknown"))
MODEL_NAME = str(metadata.get("model_name", "scoring-model"))
MODEL_KEYS = ("model_name", "model_alias", "model_version", "run_id", "onnx_sha256", "exported_at")

# --- Métriques Prometheus (Lab 7) ------------------------------------------------------
# Règle d'or : des labels à valeurs PEU nombreuses et connues à l'avance. Jamais un
# identifiant, un montant ou une valeur saisie par le client (chaque valeur distincte
# crée une nouvelle série temporelle dans Prometheus).
disable_created_metrics()  # pas de séries *_created : moitié moins de séries à stocker
# Registre propre au module (+ métriques du processus) : réimporter serve.py, comme le font
# les tests, ne provoque pas d'erreur « Duplicated timeseries in CollectorRegistry ».
REGISTRY = CollectorRegistry()
for collector in (PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR):
    REGISTRY.register(collector)
PREDICTIONS = Counter(
    "ml_predictions_total",
    "Décisions rendues (outcome=1 : classe positive de dataset.yaml, ici un refus)",
    ["model", "version", "outcome"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Durée de traitement d'une requête /predict (validation exclue)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5],
    registry=REGISTRY,
)
SCORES = Histogram(
    "ml_prediction_score",
    "Score servi : probabilité de la classe positive",
    buckets=[round(0.05 * i, 2) for i in range(1, 20)],  # 0.05, 0.10, ..., 0.95
    registry=REGISTRY,
)
GROUP_PREDICTIONS = Counter(
    "ml_predictions_by_group_total",
    "Décisions par groupe d'un attribut sensible de dataset.yaml (équité, Lab 8)",
    ["attribute", "group", "outcome"],
    registry=REGISTRY,
)
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Requêtes HTTP par route (modèle de chemin) et code de réponse",
    ["method", "handler", "code"],
    registry=REGISTRY,
)
MODEL_INFO = Info("ml_model", "Modèle servi par cette instance de l'API", registry=REGISTRY)
MODEL_INFO.info(
    {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION,
        "threshold": f"{THRESHOLD:.4f}",
    }
)

# Attributs sensibles reçus par l'API (un attribut « d'audit » absent des features n'est
# pas envoyé par le client : il ne peut pas être suivi ici).
SENSITIVE = [s for s in cfg.sensitive if s.column in cfg.features]
# Garde-fou de cardinalité : au plus N groupes par attribut et par instance, le reste
# est compté dans « other » (un client ne peut pas créer des séries à volonté).
MAX_GROUPS = int(os.getenv("MAX_GROUPS_PER_ATTRIBUTE", "20"))
_known_groups: dict[str, set[str]] = {s.column: set() for s in SENSITIVE}


def group_of(s, value: Any) -> str:
    """Groupe d'une valeur, avec le découpage du Lab 8 (SensitiveAttribute.groups).

    Réécrit sans pandas : pd.cut coûte ~0,6 ms par appel, six fois l'inférence ONNX.
    bins=[0, 25, 120] et right=False : [0, 25) -> labels[0], [25, 120) -> labels[1].
    """
    if value is None:
        return "missing"
    if s.bins is None:
        group = str(value)
    else:
        i = bisect.bisect_right(s.bins, value) - 1
        if not 0 <= i < len(s.bins) - 1:
            return "missing"  # hors des bornes : pd.cut renvoie aussi NaN
        group = s.labels[i] if s.labels else f"[{s.bins[i]}, {s.bins[i + 1]})"
    known = _known_groups[s.column]
    if group not in known and len(known) >= MAX_GROUPS:
        return "other"
    known.add(group)
    return group


# --- Journal des prédictions (une ligne JSON par demande) ---------------------------------
# Ce sont des données personnelles (demandes de crédit) : rétention courte dans Loki,
# accès restreint, désactivable avec LOG_PREDICTIONS=false.
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() in ("1", "true", "yes")
prediction_log = logging.getLogger("scoring.predictions")
if not prediction_log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    prediction_log.addHandler(_handler)
    prediction_log.setLevel(logging.INFO)
    prediction_log.propagate = False

# --- Schéma d'entrée généré depuis dataset.yaml ---------------------------------------
# strict : "12" (chaîne) est refusé pour une colonne numérique, 3 pour une catégorielle.
# null est accepté : valeur manquante, imputée par le pipeline (médiane / mode).
Numeric = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]
Categorical = Annotated[str | None, Field(strict=True, min_length=1)]
Instance = create_model(
    "Instance",
    __config__=ConfigDict(extra="forbid"),  # colonne inconnue = erreur 422
    **{c: (Numeric, ...) for c in cfg.numeric},  # ... = colonne obligatoire
    **{c: (Categorical, ...) for c in cfg.categorical},
)


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(..., min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    probability: float  # probabilité de la classe positive (target.positive_class)
    prediction: int  # 1 si probability >= threshold, sinon 0


class PredictResponse(BaseModel):
    model_version: str
    positive_class: str
    threshold: float
    predictions: list[Prediction]


app = FastAPI(title=f"Scoring API — {cfg.name}", version=API_VERSION)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Compte chaque requête par route et code HTTP (taux d'erreur, requêtes rejetées)."""
    code = 500  # une exception non gérée devient une 500
    try:
        response = await call_next(request)
        code = response.status_code
        return response
    finally:
        route = request.scope.get("route")  # renseignée par le routeur après call_next
        # Modèle de chemin (/predict), jamais l'URL brute : cardinalité bornée
        handler = getattr(route, "path", "other")
        HTTP_REQUESTS.labels(request.method, handler, str(code)).inc()


def score(rows: list[dict[str, Any]]) -> np.ndarray:
    """Probabilité de la classe positive pour chaque demande (une entrée ONNX par colonne).

    numpy plutôt que pandas : construire un DataFrame coûte ~50x plus cher que l'inférence.
    """
    # null -> NaN (numérique) ou "" (catégoriel) : imputés dans le graphe ONNX
    feed = {c: np.array([[r[c]] for r in rows], dtype=np.float64) for c in cfg.numeric}
    feed |= {c: np.array([[r[c] or ""] for r in rows], dtype=object) for c in cfg.categorical}
    (proba,) = session.run(["probabilities"], feed)
    return proba[:, 1]


def observe(rows: list[dict[str, Any]], proba: np.ndarray, decisions: list[int]) -> None:
    """Lab 7 : métriques métier + journal JSON des prédictions."""
    for outcome in (0, 1):
        n = decisions.count(outcome)
        if n:
            PREDICTIONS.labels(MODEL_NAME, MODEL_VERSION, str(outcome)).inc(n)
    for p in proba:
        SCORES.observe(float(p))
    for s in SENSITIVE:
        for row, outcome in zip(rows, decisions, strict=True):
            GROUP_PREDICTIONS.labels(s.column, group_of(s, row[s.column]), str(outcome)).inc()
    if LOG_PREDICTIONS:
        for row, p, outcome in zip(rows, proba, decisions, strict=True):
            prediction_log.info(
                json.dumps(
                    {
                        "event": "prediction",
                        "model_version": MODEL_VERSION,
                        "api_version": API_VERSION,
                        "probability": round(float(p), 6),
                        "prediction": outcome,
                        "features": row,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/metadata")
def get_metadata():
    return {
        "api_version": API_VERSION,
        "dataset": cfg.name,
        "target": cfg.target,
        "positive_class": str(cfg.positive_class),
        "threshold": THRESHOLD,
        "features": {"numeric": cfg.numeric, "categorical": cfg.categorical},
        "model": {k: metadata.get(k) for k in MODEL_KEYS},
    }


@app.get("/version")
def version():
    # Lab 5 : quelle version a répondu ? (canary, rollback, tests de fumée du Lab 6)
    return {"version": API_VERSION, "model_version": MODEL_VERSION}


@app.get("/metrics", include_in_schema=False)
def metrics():
    # Route explicite plutôt que app.mount("/metrics", make_asgi_app()) : le montage
    # répond 307 sur /metrics (redirection vers /metrics/), déroutant avec curl.
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with LATENCY.time():
        rows = [i.model_dump() for i in req.instances]
        try:
            proba = score(rows)
        except Exception as e:
            raise HTTPException(500, str(e)) from e
        decisions = [int(p >= THRESHOLD) for p in proba]
        observe(rows, proba, decisions)
        return PredictResponse(
            model_version=MODEL_VERSION,
            positive_class=str(cfg.positive_class),
            threshold=THRESHOLD,
            predictions=[
                Prediction(probability=float(p), prediction=d)
                for p, d in zip(proba, decisions, strict=True)
            ],
        )
