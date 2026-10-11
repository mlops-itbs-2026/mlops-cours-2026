# Fichier : packaging/bento/service.py
"""Étape 6.2 — variante BentoML de l'API : même route /predict, même JSON, port 3000."""

from typing import Annotated

import bentoml
import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, create_model

MODEL_TAG = "scoring_model:latest"
# Schéma lu dans les métadonnées du modèle (écrites par save_model.py depuis dataset.yaml)
META = bentoml.models.get(MODEL_TAG).info.metadata
FEATURES = [*META["numeric"], *META["categorical"]]
THRESHOLD = float(META["threshold"])

Numeric = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]
Categorical = Annotated[str | None, Field(strict=True, min_length=1)]
Instance = create_model(
    "Instance",
    __config__=ConfigDict(extra="forbid"),
    **{c: (Numeric, ...) for c in META["numeric"]},
    **{c: (Categorical, ...) for c in META["categorical"]},
)


@bentoml.service(name="scoring_bento", resources={"cpu": "1"}, traffic={"timeout": 10})
class ScoringBento:
    # Référence au model store : le modèle est embarqué dans le Bento au build.
    bento_model = bentoml.models.BentoModel(MODEL_TAG)

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.bento_model)

    @bentoml.api
    def predict(self, instances: Annotated[list[Instance], Field(min_length=1)]) -> dict:
        df = pd.DataFrame([i.model_dump() for i in instances], columns=FEATURES)
        df = df.replace({None: np.nan})  # null -> NaN : valeur manquante pour scikit-learn
        proba = self.model.predict_proba(df)[:, 1]
        return {
            "model_version": META["model_version"],
            "positive_class": META["positive_class"],
            "threshold": THRESHOLD,
            "predictions": [
                {"probability": float(p), "prediction": int(p >= THRESHOLD)} for p in proba
            ],
        }
