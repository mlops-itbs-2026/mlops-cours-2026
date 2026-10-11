# Fichier : serving/v2_client.py
"""Étape 8.3 — interroge le modèle servi par Triton (protocole V2) et décide au seuil métier.

Triton ne renvoie que des probabilités : la décision est prise ICI, avec le seuil publié dans
la configuration du modèle (paramètre decision_threshold écrit par prepare_triton.py, donc
le seuil du run d'entraînement du Lab 1). Avec --compare, les mêmes demandes sont envoyées à
l'API du Lab 4 : mêmes probabilités et mêmes décisions attendues.

Usage (racine du repo) :
    uv run python serving/v2_client.py --host "$TRITON_HOST" -n 5 --compare "$ISVC_HOST"
"""

import argparse
import json
import urllib.request

from payloads import complete_rows, load_rows, v1_body, v2_body, v2_positive_proba

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True, help="en-tête Host de l'InferenceService Triton")
parser.add_argument("--gateway", default="http://localhost:30080", help="passerelle Kourier")
parser.add_argument("--model", default="scoring-model", help="nom du modèle côté Triton")
parser.add_argument("-n", type=int, default=5, help="nombre de demandes (une seule requête)")
parser.add_argument("--compare", metavar="HOST", help="en-tête Host de l'API du Lab 4")
args = parser.parse_args()


def call(host: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Host": host, "Content-Type": "application/json"}
    req = urllib.request.Request(args.gateway + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


rows = complete_rows(load_rows())[: args.n]
params = call(args.host, f"/v2/models/{args.model}/config")["parameters"]
threshold = float(params["decision_threshold"]["string_value"])
positive = params["positive_class"]["string_value"]

proba = v2_positive_proba(call(args.host, f"/v2/models/{args.model}/infer", v2_body(rows)))
decisions = [int(p >= threshold) for p in proba]
print(f"Triton : seuil {threshold} (publié avec le modèle), classe positive « {positive} »")
for p, d in zip(proba, decisions, strict=True):
    print(f"  P({positive}) = {p:.4f} -> décision {d}")

if args.compare:
    api = call(args.compare, "/predict", v1_body(rows))
    api_proba = [x["probability"] for x in api["predictions"]]
    api_decisions = [x["prediction"] for x in api["predictions"]]
    gap = max(abs(a - b) for a, b in zip(proba, api_proba, strict=True))
    same = sum(a == b for a, b in zip(decisions, api_decisions, strict=True))
    print(f"API Lab 4 : seuil {api['threshold']} · écart max des probabilités {gap:.1e}")
    print(f"Décisions identiques : {same}/{len(rows)}")
