# Fichier : tests/conftest.py
"""Fixtures communes aux tests (Lab 6). Les tests tournent SANS MLflow, sans données DVC
et sans cluster : c'est ce que voit la CI GitHub. L'API est donc testée avec un
mini-modèle ONNX, entraîné à la volée sur des données synthétiques construites depuis
dataset.yaml (mêmes entrées et mêmes sorties que le modèle exporté au Lab 4).
"""

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
# « src » et « pipelines » importables, ainsi que l'API (app.serve, dans packaging/)
for path in (ROOT, ROOT / "packaging"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("DATASET_CONFIG", str(ROOT / "dataset.yaml"))

from src.dataset_config import load_config  # noqa: E402


@pytest.fixture(autouse=True)
def _racine_du_repo(monkeypatch):
    """Les scripts du projet lisent dataset.yaml et params.yaml depuis la racine."""
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="session")
def contract():
    return load_config(ROOT / "dataset.yaml")


def synthetic_frame(cfg, n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Données factices conformes au contrat : numériques continues, 3 modalités par colonne."""
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(50, 10, n) for c in cfg.numeric}
    data |= {c: rng.choice(["a", "b", "c"], n) for c in cfg.categorical}
    df = pd.DataFrame(data)
    signal = df[cfg.numeric[0]] if cfg.numeric else pd.Series(rng.normal(50, 10, n))
    df[cfg.target] = np.where(signal > 50, str(cfg.positive_class), "autre")
    return df


@pytest.fixture(scope="session")
def make_frame():
    """La fabrique de données synthétiques, pour les tests qui en ont besoin."""
    return synthetic_frame


@pytest.fixture(scope="session")
def mini_model(contract, tmp_path_factory) -> Path:
    """model.onnx + metadata.json : même contrat d'entrées/sorties que le modèle du Lab 4."""
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import DoubleTensorType, StringTensorType
    from skl2onnx.sklapi import CastTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cfg = contract
    X, y = cfg.split_xy(synthetic_frame(cfg))
    # Comme export_onnx.py (Lab 4) : l'imputer ONNX n'existe qu'en float32
    numeric = Pipeline(
        [
            ("scale", StandardScaler()),
            ("float32", CastTransformer(dtype=np.float32)),
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(missing_values="", strategy="constant", fill_value="?")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer([("num", numeric, cfg.numeric), ("cat", categorical, cfg.categorical)])
    model = Pipeline([("preprocess", pre), ("model", LogisticRegression())]).fit(X, y)
    types = [(c, DoubleTensorType([None, 1])) for c in cfg.numeric]
    types += [(c, StringTensorType([None, 1])) for c in cfg.categorical]
    onx = to_onnx(model, initial_types=types, target_opset=17, options={"zipmap": False})

    folder = tmp_path_factory.mktemp("model")
    (folder / "model.onnx").write_bytes(onx.SerializeToString())
    metadata = {"model_name": "scoring-model", "model_version": "test", "threshold": 0.4}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return folder / "model.onnx"


@pytest.fixture(scope="session")
def api(mini_model):
    """Client de test de l'API du Lab 4-5, servant le mini-modèle (aucun serveur lancé)."""
    from fastapi.testclient import TestClient

    patch = pytest.MonkeyPatch()
    patch.setenv("MODEL_PATH", str(mini_model))  # lu par serve.py à l'import
    patch.setenv("APP_VERSION", "9.9.9")
    patch.delenv("DECISION_THRESHOLD", raising=False)
    sys.modules.pop("app.serve", None)
    serve = importlib.import_module("app.serve")
    with TestClient(serve.app) as client:
        yield client
    patch.undo()
    sys.modules.pop("app.serve", None)
