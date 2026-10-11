# Fichier : governance/fairness_mitigation.py
"""Étape 4 — atténuer les disparités : trois familles de stratégies comparées au champion.

Pour chaque attribut sensible de dataset.yaml :
  1. « sans l'attribut » (fairness through unawareness) : le même modèle ré-entraîné sans
     la colonne sensible, plus un test de proxy : peut-on deviner le groupe à partir des
     autres variables ? (si oui, retirer la colonne ne suffit pas) ;
  2. ThresholdOptimizer (post-traitement) : un seuil par groupe sous contrainte
     « equalized odds ». Son objectif par défaut est l'accuracy ; on montre ce que cela
     coûte, puis on lui fait optimiser le coût métier de dataset.yaml ;
  3. ExponentiatedGradient (réduction, in-processing) : ré-entraînement sous la même
     contrainte, avec le coût métier comme objectif (ErrorRate pondérée).
Tout est évalué sur le jeu de test, au même titre que le champion.

Usage (racine du repo) :
    uv run python -m governance.fairness_mitigation | tee reports/fairness_mitigated.txt
Sorties : le comparatif sur la sortie standard + reports/fairness_mitigated.json
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ErrorRate, ExponentiatedGradient
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from governance.fairness_analysis import fairness_indicators, load_champion
from src.dataset_config import DatasetConfig, load_config
from src.train import TEST_CSV, TRAIN_CSV

REPORT_JSON = Path("reports/fairness_mitigated.json")


def without(cfg: DatasetConfig, column: str) -> DatasetConfig:
    """La même configuration, sans une colonne (prétraitement et features)."""
    return dataclasses.replace(
        cfg,
        numeric=[c for c in cfg.numeric if c != column],
        categorical=[c for c in cfg.categorical if c != column],
    )


def proxy_test(cfg: DatasetConfig, X: pd.DataFrame, groups: pd.Series) -> dict:
    """Peut-on retrouver le groupe sensible à partir des AUTRES variables ? (AUC en CV 5)

    AUC ~ 0,5 : aucune information ; AUC élevée : des « proxies » trahissent le groupe.
    """
    if groups.nunique() < 2:
        return {"auc": None, "top_proxies": {}}
    scoring = "roc_auc" if groups.nunique() == 2 else "roc_auc_ovr_weighted"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    def auc(features: list[str]) -> float:
        sub = dataclasses.replace(
            cfg,
            numeric=[c for c in cfg.numeric if c in features],
            categorical=[c for c in cfg.categorical if c in features],
        )
        clf = Pipeline(
            [("preprocess", sub.build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
        )
        return float(cross_val_score(clf, X[features], groups, cv=cv, scoring=scoring).mean())

    single = {f: auc([f]) for f in cfg.features}  # pouvoir révélateur de chaque variable seule
    top = dict(sorted(single.items(), key=lambda kv: kv[1], reverse=True)[:3])
    return {"auc": auc(cfg.features), "top_proxies": top}


def cost_weighted(cfg: DatasetConfig, X, y, groups):
    """Duplique les lignes pour que « accuracy » = « coût métier » (C_FN/C_FP répétitions).

    ThresholdOptimizer n'optimise que des scores prédéfinis (accuracy…) et n'accepte pas de
    poids : avec chaque mauvais payeur compté 5 fois, maximiser l'accuracy revient à
    minimiser 5 x FN + 1 x FP. Les contraintes TPR/FPR, calculées par classe, ne changent pas.
    """
    ratio = cfg.cost_false_negative / cfg.cost_false_positive
    reps = np.where(y == 1, max(round(ratio), 1), max(round(1 / ratio), 1))
    idx = X.index.repeat(reps)
    return X.loc[idx], y.loc[idx], groups.loc[idx]


def evaluate(cfg, y_true, y_pred, groups, ref_cost: float, uses_attribute: bool) -> dict:
    y_pred = np.asarray(y_pred).astype(int)
    cost = cfg.business_cost(y_true, y_pred)
    return {
        "business_cost": cost,
        "cost_change": cost / ref_cost - 1,
        "refusal_rate": float(y_pred.mean()),
        **fairness_indicators(y_true, y_pred, groups),
        "uses_attribute_at_prediction": uses_attribute,
    }


def main() -> None:
    cfg = load_config()
    model, threshold, version = load_champion()
    X_train, y_train = cfg.read_xy(TRAIN_CSV)
    X_test, y_test = cfg.read_xy(TEST_CSV)
    s_train, s_test = cfg.sensitive_frame(X_train), cfg.sensitive_frame(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    base_pred = (proba >= threshold).astype(int)
    ref_cost = cfg.business_cost(y_test, base_pred)
    costs = {"fn": cfg.cost_false_negative, "fp": cfg.cost_false_positive}
    last_step = model.steps[-1][0]  # « model » : nom de l'estimateur dans le Pipeline (Lab 1)
    print(f"Champion v{version} · seuil {threshold:.3f} · coût de référence {ref_cost:.0f}")

    report = {"model_version": version, "decision_threshold": threshold, "attributes": {}}
    for attr, group_col in zip(cfg.sensitive, s_test.columns, strict=True):
        g_train, g_test = s_train[group_col], s_test[group_col]
        decisions = {"champion": (base_pred, True)}  # nom -> (décisions, attribut en prod ?)

        # 1. Fairness through unawareness : on retire la colonne et on ré-entraîne
        cfg_wo = without(cfg, attr.column)
        # « preprocess » : nom de l'étape de prétraitement dans le Pipeline du Lab 1
        unaware = clone(model).set_params(preprocess=cfg_wo.build_preprocessor())
        unaware.fit(X_train[cfg_wo.features], y_train)
        p_unaware = unaware.predict_proba(X_test[cfg_wo.features])[:, 1]
        decisions["sans_attribut"] = (p_unaware >= threshold, False)
        proxies = proxy_test(cfg_wo, X_train, g_train)

        # 2. Post-traitement : un seuil par groupe (le modèle n'est pas ré-entraîné)
        for objective, (Xf, yf, gf) in {
            "accuracy": (X_train, y_train, g_train),
            "cout": cost_weighted(cfg, X_train, y_train, g_train),
        }.items():
            to = ThresholdOptimizer(
                estimator=model,
                constraints="equalized_odds",
                objective="accuracy_score",
                prefit=True,
                predict_method="predict_proba",
            )
            to.fit(Xf, yf, sensitive_features=gf)
            y_to = to.predict(X_test, sensitive_features=g_test, random_state=cfg.random_state)
            decisions[f"threshold_optimizer_{objective}"] = (y_to, True)

        # 3. In-processing : ré-entraînement sous contrainte, objectif = coût métier.
        # Les poids de classes du champion sont neutralisés : les coûts les remplacent.
        base = clone(model)
        if f"{last_step}__class_weight" in base.get_params():
            base.set_params(**{f"{last_step}__class_weight": None})
        eg = ExponentiatedGradient(
            base,
            constraints=EqualizedOdds(),
            objective=ErrorRate(costs=costs),
            sample_weight_name=f"{last_step}__sample_weight",  # poids routés vers le modèle
        )
        eg.fit(X_train, y_train, sensitive_features=g_train)
        y_eg = eg.predict(X_test, random_state=cfg.random_state)  # classifieur randomisé
        decisions["exponentiated_gradient"] = (y_eg, False)

        results = {
            name: evaluate(cfg, y_test, y_pred, g_test, ref_cost, uses_attribute)
            for name, (y_pred, uses_attribute) in decisions.items()
        }
        table = pd.DataFrame(results).T
        shown = pd.DataFrame(
            {
                "coût": table["business_cost"].map("{:.0f}".format),
                "Δcoût": table["cost_change"].map("{:+.1%}".format),
                "refus": table["refusal_rate"].map("{:.0%}".format),
                "DP diff": table["demographic_parity_difference"].map("{:.3f}".format),
                "ratio acc.": table["acceptance_rate_ratio"].map("{:.2f}".format),
                "EO diff": table["equalized_odds_difference"].map("{:.3f}".format),
                "attribut en prod": table["uses_attribute_at_prediction"].map(
                    {True: "oui", False: "non"}
                ),
            }
        )
        print(f"\n=== Attribut sensible : {attr.column} ===")
        print(shown.to_string())
        auc = proxies["auc"]
        if auc is not None:
            top = ", ".join(f"{f} ({a:.2f})" for f, a in proxies["top_proxies"].items())
            print(f"Proxy : le groupe se devine sans la colonne avec une AUC de {auc:.2f}")
            print(f"        variables les plus révélatrices : {top}")
        report["attributes"][attr.column] = {"proxy": proxies, "strategies": results}

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")
    print(f"\n-> {REPORT_JSON}")


if __name__ == "__main__":
    main()
