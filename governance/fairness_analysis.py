# Fichier : governance/fairness_analysis.py
"""Étape 3 — mesure des disparités du modèle champion, attribut sensible par attribut sensible.

Pour CHAQUE attribut de la section `sensitive` de dataset.yaml (aucun nom en dur ici) :
  - métriques par groupe au seuil de décision du modèle (param `decision_threshold` du
    run, Lab 1) : taux de défaut réel, taux de refus, TPR, FPR, coût métier par demande ;
  - indicateurs d'équité : parité démographique (écart, ratio), ratio d'acceptation
    (règle des 4/5), equalized odds ;
  - intervalle de confiance bootstrap du taux de refus (petits groupes = chiffres fragiles).
Classe positive = « à risque » : pour Fairlearn, « sélectionner » une demande = la REFUSER.

Usage (racine du repo) :
    uv run python -m governance.fairness_analysis | tee reports/fairness_baseline.txt
Sorties : le rapport sur la sortie standard + reports/fairness_baseline.json
"""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    false_positive_rate_difference,
    selection_rate,
    true_positive_rate,
)
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from src.dataset_config import DatasetConfig, load_config
from src.promote import MODEL_NAME
from src.train import TEST_CSV

REPORT_JSON = Path("reports/fairness_baseline.json")
MIN_GROUP_SIZE = 30  # en dessous, les taux d'un groupe sont trop incertains pour conclure
N_BOOT = 200  # tirages bootstrap pour l'intervalle de confiance à 95 %
FOUR_FIFTHS = 0.8  # règle des 4/5 : ratio minimal toléré entre taux d'acceptation
RATES = ["taux_defaut", "taux_refus", "TPR", "FPR"]


def load_champion() -> tuple[Pipeline, float, str]:
    """Le champion du Registry, son seuil de décision (Lab 1) et son numéro de version."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    # Le seuil fait partie du modèle : c'est lui qui décide (et non 0,5)
    threshold = float(client.get_run(version.run_id).data.params["decision_threshold"])
    # On charge la version exacte : l'alias pourrait bouger pendant l'audit
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")
    return model, threshold, str(version.version)


def group_table(cfg: DatasetConfig, y_true, y_pred, groups: pd.Series) -> pd.DataFrame:
    """Une ligne par groupe : effectif, taux réels, taux de refus, TPR, FPR, coût moyen."""
    metrics = {
        "n": count,
        "taux_defaut": lambda yt, yp: float(np.mean(yt)),  # mauvais payeurs réels
        "taux_refus": selection_rate,  # P(refus) : la « sélection » de Fairlearn
        "TPR": true_positive_rate,  # mauvais payeurs effectivement refusés
        "FPR": false_positive_rate,  # BONS payeurs refusés à tort
        "cout_moyen": lambda yt, yp: cfg.business_cost(yt, yp) / len(yt),
    }
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=groups)
    table = mf.by_group
    table.loc["(ensemble)"] = mf.overall
    # IC à 95 % du taux de refus : rééchantillonnage des demandes (bootstrap)
    boot = MetricFrame(
        metrics={"refus": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        n_boot=N_BOOT,
        ci_quantiles=[0.025, 0.975],
        random_state=cfg.random_state,
    )
    low, high = boot.by_group_ci
    table["refus_ic95_bas"] = low["refus"]
    table["refus_ic95_haut"] = high["refus"]
    return table


def fairness_indicators(y_true, y_pred, groups: pd.Series) -> dict[str, float]:
    """Indicateurs d'équité d'un jeu de décisions (0/1, 1 = refus) pour un attribut."""
    y_pred = np.asarray(y_pred)
    return {
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=groups
        ),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_true, y_pred, sensitive_features=groups
        ),
        # Règle des 4/5 : elle porte sur l'issue FAVORABLE (l'acceptation = 1 - refus)
        "acceptance_rate_ratio": demographic_parity_ratio(
            y_true, 1 - y_pred, sensitive_features=groups
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=groups
        ),
        "false_positive_rate_difference": false_positive_rate_difference(
            y_true, y_pred, sensitive_features=groups
        ),
    }


def print_attribute(column: str, table: pd.DataFrame, ind: dict) -> None:
    shown = table.copy()
    shown["n"] = shown["n"].astype(int)
    for col in RATES:
        shown[col] = shown[col].map("{:.0%}".format)
    shown["cout_moyen"] = shown["cout_moyen"].map("{:.2f}".format)
    shown["refus_IC95"] = [
        f"[{lo:.0%} ; {hi:.0%}]" if pd.notna(lo) else ""
        for lo, hi in zip(table["refus_ic95_bas"], table["refus_ic95_haut"], strict=True)
    ]
    print(f"\n=== Attribut sensible : {column} ===")
    print(shown.drop(columns=["refus_ic95_bas", "refus_ic95_haut"]).to_string())
    ratio = ind["acceptance_rate_ratio"]
    alert = "  <-- sous la règle des 4/5" if ratio < FOUR_FIFTHS else ""
    print(f"Demographic Parity Diff (refus)  : {ind['demographic_parity_difference']:.3f}")
    print(f"Demographic Parity Ratio (refus) : {ind['demographic_parity_ratio']:.3f}")
    print(f"Ratio d'acceptation (4/5)        : {ratio:.3f}{alert}")
    print(f"Equalized Odds Diff (TPR/FPR)    : {ind['equalized_odds_difference']:.3f}")
    print(f"Groupe le plus refusé            : {ind['most_refused_group']}")
    if ind["small_groups"]:
        print(f"Attention : moins de {MIN_GROUP_SIZE} demandes pour {ind['small_groups']}")


def main() -> None:
    cfg = load_config()
    model, threshold, version = load_champion()
    X_test, y_test = cfg.read_xy(TEST_CSV)
    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    sensitive = cfg.sensitive_frame(X_test)  # un groupe par attribut (age découpé en tranches)

    print(f"Modèle {MODEL_NAME} v{version} · seuil de décision {threshold:.3f}")
    print(f"{len(y_test)} demandes de test · prédiction 1 = « {cfg.positive_class} » = REFUS")
    cost = cfg.business_cost(y_test, y_pred)
    print(f"Global : refus {y_pred.mean():.0%}, coût métier {cost:.0f}")

    report = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "decision_threshold": threshold,
        "n_test": len(y_test),
        "business_cost": cost,
        "refusal_rate": float(y_pred.mean()),
        "attributes": {},
    }
    for attr, group_col in zip(cfg.sensitive, sensitive.columns, strict=True):
        groups = sensitive[group_col]
        table = group_table(cfg, y_test, y_pred, groups)
        by_group = table.drop("(ensemble)")
        ind = fairness_indicators(y_test, y_pred, groups) | {
            "most_refused_group": by_group["taux_refus"].idxmax(),
            "small_groups": [g for g, n in by_group["n"].items() if n < MIN_GROUP_SIZE],
        }
        print_attribute(attr.column, table, ind)
        report["attributes"][attr.column] = {
            "groups_column": group_col,
            "groups": by_group.to_dict(orient="index"),
            **ind,
        }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")
    print(f"\n-> {REPORT_JSON}")


if __name__ == "__main__":
    main()
