# Fichier : tests/test_model_behavior.py
"""Tests comportementaux du modèle (Lab 3), à la manière de CheckList (Ribeiro et al., 2020).

On teste ce que fait le modèle, pas seulement sa métrique globale :
- performance minimale (MFT) : métrique primaire au seuil de dataset.yaml, coût métier
  inférieur à celui des politiques naïves, performance par sous-groupe sensible ;
- invariance (INV) : changer un attribut sans effet attendu ne change pas la décision ;
- directionnels (DIR) : augmenter une variable ne déplace le risque que dans le bon sens.
Modèle testé : models/model (dvc repro, ou step train du pipeline), sur data/processed/test.csv,
au seuil de décision du modèle. Les attentes propres au dataset sont dans tests/behavior.yaml.
Sans modèle local (CI du Lab 6), les tests sont ignorés.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset_config import load_config
from src.train import MODEL_DIR, TEST_CSV, compute_metrics

CFG = load_config()
EXPECT = yaml.safe_load(Path(__file__).with_name("behavior.yaml").read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "MLmodel").exists() or not TEST_CSV.exists(),
    reason="pas de modèle local : dvc pull ou dvc repro (Lab 1)",
)


@pytest.fixture(scope="module")
def model():
    """Le modèle (Pipeline scikit-learn) et son seuil de décision (métadonnées MLflow)."""
    import mlflow
    from mlflow.models import Model

    threshold = Model.load(str(MODEL_DIR)).metadata["decision_threshold"]
    return mlflow.sklearn.load_model(str(MODEL_DIR)), threshold


@pytest.fixture(scope="module")
def data():
    """Features, cible 0/1 et fichier brut du jeu de test (pour les groupes sensibles)."""
    X, y = CFG.read_xy(TEST_CSV)
    return X, y, pd.read_csv(TEST_CSV)


def risk(model, X: pd.DataFrame) -> np.ndarray:
    """Le risque prédit : probabilité de la classe positive de dataset.yaml."""
    return model[0].predict_proba(X)[:, 1]


# --- Performance minimale (MFT) -------------------------------------------------------------
def test_metrique_primaire_au_seuil(model, data):
    X, y, _ = data
    value = compute_metrics(CFG, y, risk(model, X), model[1])[CFG.primary_metric]
    assert value >= CFG.promotion_threshold, (
        f"{CFG.primary_metric} = {value:.3f} < {CFG.promotion_threshold}"
    )


def test_cout_metier_inferieur_aux_politiques_naives(model, data):
    """Mieux que « tout accorder » et que « tout refuser » (matrice de coûts)."""
    X, y, _ = data
    cost = CFG.business_cost(y, risk(model, X) >= model[1])
    naive = min(CFG.business_cost(y, np.zeros(len(y))), CFG.business_cost(y, np.ones(len(y))))
    assert cost < naive, f"coût {cost:.0f} >= coût de la meilleure politique naïve {naive:.0f}"


@pytest.mark.parametrize("attribute", CFG.sensitive, ids=lambda s: s.column)
def test_performance_par_sous_groupe(model, data, attribute):
    X, y, raw = data
    proba, groups = risk(model, X), attribute.groups(raw).to_numpy()
    scores = {}
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() >= EXPECT["min_group_size"] and y[mask].nunique() == 2:
            metrics = compute_metrics(CFG, y[mask], proba[mask], model[1])
            scores[group] = round(metrics[CFG.primary_metric], 3)
    if not scores:
        pytest.skip(f"aucun groupe d'au moins {EXPECT['min_group_size']} demandes")
    below = {g: s for g, s in scores.items() if s < CFG.promotion_threshold}
    assert not below, f"{CFG.primary_metric} < {CFG.promotion_threshold} : {below}"


# --- Invariance (INV) -----------------------------------------------------------------------
@pytest.mark.parametrize("case", EXPECT.get("invariance") or [], ids=lambda c: c["column"])
def test_invariance(model, data, case):
    X, _, _ = data
    col, decision = case["column"], risk(model, X) >= model[1]
    flips = changed = 0
    for value in X[col].unique():  # chaque demande reçoit chacune des autres valeurs
        modified = X.assign(**{col: value})
        mask = (X[col] != value).to_numpy()
        flips += int(((risk(model, modified) >= model[1]) != decision)[mask].sum())
        changed += int(mask.sum())
    rate = flips / changed
    assert rate <= case["max_flip_rate"], f"{rate:.1%} des décisions changent avec {col}"


# --- Directionnels (DIR) --------------------------------------------------------------------
@pytest.mark.parametrize("case", EXPECT.get("directional") or [], ids=lambda c: c["column"])
def test_directionnel(model, data, case):
    X, _, _ = data
    modified = X.assign(**{case["column"]: X[case["column"]] * case["factor"]})
    delta = risk(model, modified) - risk(model, X)
    sign = 1 if case["expect"] == "increase" else -1
    violations = float(np.mean(sign * delta < -EXPECT["tolerance"]))
    assert violations <= EXPECT["max_violation_rate"], (
        f"{violations:.1%} des demandes : risque dans le mauvais sens"
    )
