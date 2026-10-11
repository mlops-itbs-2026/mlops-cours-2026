# Fichier : serving/payloads.py
"""Corps de requête construits à partir de dataset.yaml et de vraies lignes de test.

Deux protocoles d'inférence :
- V1 « instances » (API FastAPI du Lab 4, BentoML) : {"instances": [{colonne: valeur}, …]} ;
- V2 / Open Inference Protocol (Triton) : un tenseur [N, 1] par colonne, nommé comme elle,
  FP64 pour les numériques et BYTES (texte) pour les catégorielles : les entrées du graphe
  ONNX du Lab 4.
Aucune colonne n'est écrite ici : changer de dataset ne demande aucune modification.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # racine du repo
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

cfg = load_config(ROOT / "dataset.yaml")


def load_rows(csv: Path = ROOT / "data/processed/test.csv") -> list[dict]:
    """Lignes réelles (features de dataset.yaml seulement) ; NaN -> None (null en JSON)."""
    df = pd.read_csv(csv)[cfg.features]
    return df.astype(object).where(df.notna(), None).to_dict("records")


def complete_rows(rows: list[dict]) -> list[dict]:
    """Lignes sans valeur numérique manquante : le JSON du protocole V2 n'a pas de NaN."""
    return [r for r in rows if all(r[c] is not None for c in cfg.numeric)]


def v1_body(rows: list[dict]) -> dict:
    return {"instances": rows}


def v2_body(rows: list[dict]) -> dict:
    """Une entrée par colonne, de forme [N, 1] ; on ne demande que les probabilités."""
    n = len(rows)
    inputs = [
        {"name": c, "shape": [n, 1], "datatype": "FP64", "data": [float(r[c]) for r in rows]}
        for c in cfg.numeric
    ]
    inputs += [  # None -> "" : la valeur manquante du graphe ONNX (imputée par le pipeline)
        {"name": c, "shape": [n, 1], "datatype": "BYTES", "data": [r[c] or "" for r in rows]}
        for c in cfg.categorical
    ]
    return {"inputs": inputs, "outputs": [{"name": "probabilities"}]}


def v2_positive_proba(response: dict) -> list[float]:
    """Probabilité de la classe positive (colonne 1 du tenseur [N, 2] « probabilities »)."""
    out = next(o for o in response["outputs"] if o["name"] == "probabilities")
    return [float(p) for p in out["data"][1::2]]  # données à plat : p0, p1, p0, p1, …
