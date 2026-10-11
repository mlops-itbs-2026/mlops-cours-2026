# Fichier : governance/explain.py
"""Étape 5.1 — explications SHAP globales du modèle champion.

Le champion est un Pipeline (préprocesseur de dataset.yaml + modèle). SHAP explique le
modèle final sur les colonnes produites par le préprocesseur (après one-hot), puis on
additionne les contributions des colonnes one-hot de chaque variable d'origine (les
valeurs de Shapley sont additives) pour parler la langue du métier : « checking_status »
plutôt que « cat__checking_status_<0 ».
Algorithme selon le modèle : LinearExplainer pour une régression logistique (contributions
en log-odds), TreeExplainer pour une forêt ou un boosting (calcul exact sur les arbres) ;
tout autre modèle est expliqué en « boîte noire » (plus lent).

Usage (racine du repo) : uv run python -m governance.explain
Sorties : reports/shap_summary_bar.png, reports/shap_beeswarm.png,
          reports/shap_importance.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from governance.fairness_analysis import load_champion
from src.dataset_config import DatasetConfig, load_config
from src.train import TEST_CSV, TRAIN_CSV

REPORTS = Path("reports")
N_BACKGROUND = 100  # demandes de référence (« le demandeur moyen ») tirées du train


def origin_of_columns(model: Pipeline) -> tuple[list[str], list[str]]:
    """Pour chaque colonne en sortie du préprocesseur : son nom lisible et sa variable d'origine."""
    labels, origins = [], []
    for _, transformer, columns in model.named_steps["preprocess"].transformers_:
        if transformer == "drop":
            continue
        for out in transformer.get_feature_names_out(columns):
            src = max((c for c in columns if out == c or out.startswith(f"{c}_")), key=len)
            labels.append(src if out == src else f"{src}={out[len(src) + 1 :]}")
            origins.append(src)
    return labels, origins


def explain(cfg: DatasetConfig, model: Pipeline, X: pd.DataFrame, X_background: pd.DataFrame):
    """Valeurs SHAP de la classe positive : (par colonne one-hot, par variable d'origine)."""
    labels, origins = origin_of_columns(model)
    transform = model[:-1]  # tout le Pipeline sauf le modèle final
    background = pd.DataFrame(transform.transform(X_background), columns=labels)
    data = pd.DataFrame(transform.transform(X), columns=labels)
    masker = shap.maskers.Independent(background.sample(N_BACKGROUND, random_state=0))
    estimator = model[-1]
    if hasattr(estimator, "coef_"):  # modèle linéaire : contributions exactes en log-odds
        explainer = shap.LinearExplainer(estimator, masker)
    else:
        try:  # forêt, boosting : exact, calculé sur les chemins des arbres
            explainer = shap.TreeExplainer(estimator)
        except Exception:  # autre modèle : « boîte noire » sur P(positif), plus lent
            explainer = shap.Explainer(lambda d: estimator.predict_proba(d)[:, 1], masker)
    sv = explainer(data)
    if sv.values.ndim == 3:  # (n, colonnes, classes) pour certains modèles : classe positive
        sv = sv[..., 1]
    # Matrice colonne one-hot -> variable d'origine : la somme des contributions
    to_feature = np.array([[o == f for f in cfg.features] for o in origins], dtype=float)
    by_feature = shap.Explanation(
        values=sv.values @ to_feature,
        base_values=sv.base_values,
        data=X[cfg.features].to_numpy(dtype=object),
        feature_names=cfg.features,
    )
    return sv, by_feature, type(explainer).__name__


def output_space(model: Pipeline, X: pd.DataFrame, sv: shap.Explanation) -> str:
    """Les contributions s'additionnent-elles en probabilité ou en log-odds ?"""
    f = sv.base_values + sv.values.sum(axis=1)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return "probabilité" if np.allclose(f, p, atol=0.02) else "log-odds"


def main() -> None:
    plt.switch_backend("agg")  # pas de fenêtre : on écrit des PNG
    cfg = load_config()
    model, threshold, version = load_champion()
    X_train, _ = cfg.read_xy(TRAIN_CSV)
    X_test, _ = cfg.read_xy(TEST_CSV)
    sv, by_feature, algo = explain(cfg, model, X_test, X_train)
    space = output_space(model, X_test, by_feature)
    print(f"Champion v{version} ({type(model[-1]).__name__}) · {algo} · contributions en {space}")

    REPORTS.mkdir(exist_ok=True)
    # Importance globale = contribution absolue moyenne, par variable d'origine
    plt.figure()
    shap.plots.bar(by_feature, max_display=min(20, len(cfg.features)), show=False)
    plt.title("Importance globale (|SHAP| moyen par variable)")
    plt.savefig(REPORTS / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    # Beeswarm : sens de l'effet de chaque colonne (rouge = valeur élevée / modalité présente)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title(f"Effet de chaque colonne sur le risque ({space})")
    plt.savefig(REPORTS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.Series(np.abs(by_feature.values).mean(axis=0), index=cfg.features)
    importance = importance.sort_values(ascending=False)
    ranks = {f: i + 1 for i, f in enumerate(importance.index)}
    print("Importance globale (|SHAP| moyen) :")
    for feature, value in importance.items():
        flag = "  <-- attribut sensible" if feature in {s.column for s in cfg.sensitive} else ""
        print(f"  {ranks[feature]:>2}. {feature:<24} {value:.3f}{flag}")

    report = {
        "model_version": version,
        "estimator": type(model[-1]).__name__,
        "explainer": algo,
        "output_space": space,
        "decision_threshold": threshold,
        "importance": importance.round(4).to_dict(),
        "sensitive_ranks": {s.column: ranks[s.column] for s in cfg.sensitive},
    }
    (REPORTS / "shap_importance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"-> {REPORTS}/shap_summary_bar.png, shap_beeswarm.png, shap_importance.json")


if __name__ == "__main__":
    main()
