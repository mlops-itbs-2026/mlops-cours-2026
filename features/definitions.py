# Fichier : features/definitions.py
"""Définitions Feast du projet : construites depuis dataset.yaml (aucune colonne en dur).

`feast apply` (lancé dans features/) importe ce fichier et enregistre dans le registre :
  - l'entité « request » (une demande, clé = params.yaml > prepare.id_column) ;
  - la source hors ligne data/processed/features.parquet (produite par src/prepare.py) ;
  - la vue « request_features » (toutes les features de dataset.yaml) ;
  - le service « scoring_v1 » (l'ensemble de features consommé par le modèle).
"""

import sys
from datetime import timedelta
from pathlib import Path

import yaml
from feast import Entity, FeatureService, FeatureView, Field, FileSource, Project, ValueType
from feast.types import Float64, String

ROOT = Path(__file__).resolve().parents[1]  # racine du repo
sys.path.insert(0, str(ROOT))  # pour importer src.dataset_config depuis features/
from src.dataset_config import load_config  # noqa: E402

cfg = load_config(ROOT / "dataset.yaml")
id_column = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["prepare"][
    "id_column"
]

project = Project(name="scoring", description=f"Features du projet de scoring ({cfg.name})")

# Entité = la « clé primaire » des features : une demande de crédit
request = Entity(
    name="request",
    join_keys=[id_column],
    value_type=ValueType.INT64,
    description="Une demande de crédit (identifiant créé par src/prepare.py)",
)

# Source hors ligne : le Parquet écrit par src/prepare.py (sortie DVC)
request_source = FileSource(
    name="request_source",
    path=str(ROOT / "data" / "processed" / "features.parquet"),
    timestamp_field="event_timestamp",
)

# Vue de features : schéma déduit de dataset.yaml + source + durée de validité (ttl)
request_features = FeatureView(
    name="request_features",
    entities=[request],
    ttl=timedelta(days=3650),  # une demande reste valide 10 ans (jointure temporelle)
    schema=[Field(name=c, dtype=Float64) for c in cfg.numeric]
    + [Field(name=c, dtype=String) for c in cfg.categorical],
    source=request_source,
    online=True,
    tags={"dataset": cfg.name, "owner": "mlops-cours"},
)

# Service = la liste de features consommée par le modèle scoring-model
scoring_v1 = FeatureService(name="scoring_v1", features=[request_features])
