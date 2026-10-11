# Fichier : serving/locustfile.py
"""Test de charge de l'InferenceService (Lab 5) — rejoue de vraies lignes de test.csv.

Variables d'environnement (toutes optionnelles) :
  ISVC_HOST   en-tête Host, indispensable derrière Kourier (port 30080) : c'est lui qui
              désigne le service, ex. scoring-model-predictor.scoring.example.com
  API_FORMAT  v1 (défaut) : POST /predict, corps {"instances": [...]} (Lab 4, BentoML)
              v2          : POST /v2/models/<V2_MODEL>/infer, protocole V2 (Triton)
  V2_MODEL    nom du modèle côté Triton (défaut : scoring-model)
  BATCH_SIZE  demandes par requête (défaut : 1 ; voir le bonus « batch » de l'étape 6)

Usage (racine du repo) :
    ISVC_HOST=... uv run locust -f serving/locustfile.py --host http://localhost:30080 \
        --users 100 --spawn-rate 20 --run-time 2m --headless
"""

import os
import random

from locust import HttpUser, between, task
from payloads import complete_rows, load_rows, v1_body, v2_body

ISVC_HOST = os.getenv("ISVC_HOST", "")
API_FORMAT = os.getenv("API_FORMAT", "v1")
V2_MODEL = os.getenv("V2_MODEL", "scoring-model")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))

ROWS = load_rows() if API_FORMAT == "v1" else complete_rows(load_rows())


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.3)

    def on_start(self):
        if ISVC_HOST:
            self.client.headers["Host"] = ISVC_HOST  # routage Kourier/Knative par nom d'hôte

    @task
    def predict(self):
        rows = random.choices(ROWS, k=BATCH_SIZE)
        if API_FORMAT == "v2":
            self.client.post(
                f"/v2/models/{V2_MODEL}/infer", json=v2_body(rows), name="/v2/models/[model]/infer"
            )
        else:
            self.client.post("/predict", json=v1_body(rows))
