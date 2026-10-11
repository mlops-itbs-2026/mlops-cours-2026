# Fichier : packaging/locustfile.py
"""Test de charge de l'API (étape 5.2) — rejoue des lignes réelles de data/processed/test.csv.

Usage (racine du repo) :
    uv run locust -f packaging/locustfile.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 30s --headless
"""

import random
import sys
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

cfg = load_config(ROOT / "dataset.yaml")
test = pd.read_csv(ROOT / "data/processed/test.csv")[cfg.features]
# NaN -> None : JSON n'a pas de NaN, l'API attend null pour une valeur manquante
ROWS = test.astype(object).where(test.notna(), None).to_dict("records")


class PredictUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def predict(self):
        self.client.post("/predict", json={"instances": [random.choice(ROWS)]})
