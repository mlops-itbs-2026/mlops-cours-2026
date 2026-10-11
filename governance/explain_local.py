# Fichier : governance/explain_local.py
"""Étape 5.2 — « Pourquoi cette demande est-elle refusée ? » (explications locales SHAP).

Trois demandes du jeu de test (le refus le plus net, l'acceptation la plus nette et le cas
le plus proche du seuil), ou celle passée par --id : un waterfall SHAP par variable
d'origine, avec le seuil de décision, et les motifs principaux rédigés en clair — la base
de l'explication due au demandeur (RGPD art. 22 et 15, AI Act art. 86).

Usage (racine du repo) :
    uv run python -m governance.explain_local            # les 3 cas types
    uv run python -m governance.explain_local --id 819   # une demande précise
Sorties : reports/shap_waterfall_<cas>.png + reports/explanations_local.md
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from scipy.special import logit

from governance.explain import REPORTS, explain, output_space
from governance.fairness_analysis import load_champion
from src.dataset_config import DatasetConfig, load_config
from src.train import TEST_CSV, TRAIN_CSV

N_REASONS = 3  # nombre de motifs rédigés par décision


def pick_cases(proba: np.ndarray, threshold: float) -> dict[str, int]:
    """Positions de 3 demandes types : refus net, acceptation nette, cas limite."""
    refused = np.flatnonzero(proba >= threshold)
    accepted = np.flatnonzero(proba < threshold)
    cases = {}
    if len(refused):
        cases["refus_net"] = int(refused[np.argmax(proba[refused])])
    if len(accepted):
        cases["acceptation_nette"] = int(accepted[np.argmin(proba[accepted])])
    cases["cas_limite"] = int(np.argmin(np.abs(proba - threshold)))
    return cases


def reasons(cfg: DatasetConfig, contrib: shap.Explanation) -> tuple[list[str], list[str]]:
    """Variables qui poussent le plus vers le refus / vers l'acceptation, en clair."""
    order = np.argsort(contrib.values)
    sensitive = {s.column for s in cfg.sensitive}

    def phrase(i: int) -> str:
        name, value = contrib.feature_names[i], contrib.data[i]
        if isinstance(value, float) and value.is_integer():
            value = int(value)  # 48.0 -> 48
        flag = " [attribut sensible]" if name in sensitive else ""
        return f"{name} = {value} ({contrib.values[i]:+.2f}){flag}"

    against = [phrase(i) for i in order[::-1][:N_REASONS] if contrib.values[i] > 0]
    favour = [phrase(i) for i in order[:N_REASONS] if contrib.values[i] < 0]
    return against, favour


def counterfactual_flips(cfg, model, X_train, row: pd.DataFrame, threshold: float) -> list[str]:
    """Groupes sensibles qui feraient BASCULER la décision, toutes autres variables égales."""
    refused = model.predict_proba(row)[0, 1] >= threshold
    flips = []
    for attr in cfg.sensitive:
        train_groups, own = attr.groups(X_train), attr.groups(row).iloc[0]
        for group in sorted(set(train_groups) - {own}):
            values = X_train.loc[train_groups == group, attr.column]
            # valeur représentative du groupe : médiane (tranche d'âge…) ou la modalité
            alt = row.assign(**{attr.column: values.median() if attr.bins else values.iloc[0]})
            if (model.predict_proba(alt)[0, 1] >= threshold) != refused:
                flips.append(f"{attr.column} -> {group}")
    return flips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", type=int, help="identifiant de la demande (jeu de test)")
    args = parser.parse_args()
    plt.switch_backend("agg")

    cfg = load_config()
    id_col = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["prepare"]["id_column"]
    model, threshold, version = load_champion()
    X_train, _ = cfg.read_xy(TRAIN_CSV)
    X_test, _ = cfg.read_xy(TEST_CSV)
    ids = pd.read_csv(TEST_CSV, usecols=[id_col])[id_col].to_numpy()
    proba = model.predict_proba(X_test)[:, 1]

    if args.id is not None:
        if args.id not in ids:
            raise SystemExit(f"Demande {args.id} absente de {TEST_CSV}")
        cases = {f"demande_{args.id}": int(np.flatnonzero(ids == args.id)[0])}
    else:
        cases = pick_cases(proba, threshold)
    rows = X_test.iloc[list(cases.values())]
    _, by_feature, _ = explain(cfg, model, rows, X_train)
    space = output_space(model, rows, by_feature)
    # Le seuil, exprimé dans l'unité des contributions (log-odds pour une régression logistique)
    cut = threshold if space == "probabilité" else float(logit(threshold))

    REPORTS.mkdir(exist_ok=True)
    lines = [f"# Explications individuelles — {cfg.name}, modèle v{version}", ""]
    lines += [f"Seuil de refus : P({cfg.positive_class}) >= {threshold:.3f}.", ""]
    for k, (case, pos) in enumerate(cases.items()):
        contrib = by_feature[k]
        decision = "REFUS" if proba[pos] >= threshold else "ACCORD"
        against, favour = reasons(cfg, contrib)
        flips = counterfactual_flips(cfg, model, X_train, X_test.iloc[[pos]], threshold)
        flip_text = ", ".join(flips) or "aucun"
        p = proba[pos]
        title = f"{case} — demande {ids[pos]} : P({cfg.positive_class}) = {p:.3f} → {decision}"
        print(f"\n{title}")
        for label, items in (("vers le refus", against), ("vers l'accord", favour)):
            print(f"  {label:<14}: " + ("\n" + " " * 18).join(items or ["—"]))
        print(f"  bascule si l'on change seulement : {flip_text}")
        lines += [f"## {title}", "", "Principaux motifs défavorables :", ""]
        lines += [f"- {r}" for r in against] or ["- aucun"]
        lines += ["", "Principaux éléments favorables :", ""]
        lines += [f"- {r}" for r in favour] or ["- aucun"]
        lines += ["", f"Test contrefactuel (un seul attribut sensible modifié) : {flip_text}."]
        lines += ["", f"![waterfall](shap_waterfall_{case}.png)", ""]

        plt.figure()
        shap.plots.waterfall(contrib, max_display=10, show=False)
        plt.axvline(cut, color="black", linestyle="--", linewidth=1)
        plt.title(f"{case} : {decision} (seuil en pointillés, {space})")
        plt.savefig(REPORTS / f"shap_waterfall_{case}.png", dpi=150, bbox_inches="tight")
        plt.close()

    (REPORTS / "explanations_local.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n-> {len(cases)} waterfall(s) + {REPORTS}/explanations_local.md")


if __name__ == "__main__":
    main()
