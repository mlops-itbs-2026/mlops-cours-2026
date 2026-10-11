# Fichier : packaging/sample_request.py
"""Construit un corps JSON pour POST /predict à partir de vraies lignes de test.

Usage (racine du repo) :
    uv run python packaging/sample_request.py > /tmp/request.json        # 1 demande
    uv run python packaging/sample_request.py -n 3 > /tmp/batch.json     # 3 demandes
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1, help="nombre de demandes")
parser.add_argument("--csv", default=str(ROOT / "data/processed/test.csv"))
args = parser.parse_args()

cfg = load_config(ROOT / "dataset.yaml")
rows = pd.read_csv(args.csv)[cfg.features].head(args.n)
rows = rows.astype(object).where(rows.notna(), None)  # NaN -> null (JSON n'a pas de NaN)
print(json.dumps({"instances": rows.to_dict("records")}, ensure_ascii=False))
