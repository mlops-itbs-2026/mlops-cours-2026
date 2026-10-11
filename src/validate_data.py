# Fichier : src/validate_data.py
"""Valide un fichier de données brutes contre le contrat dataset.yaml.

Gate du pipeline (Lab 2) : si un contrôle échoue, le script sort avec le code 1 et
l'orchestrateur (DVC, Kubeflow, Airflow, CI) s'arrête AVANT l'entraînement. Les contrôles
sont des contrôles de schéma simples, tous déduits de dataset.yaml (aucun nom de colonne
en dur). Le Lab 3 les remplacera par une suite Great Expectations, sans changer
l'interface :

    python -m src.validate_data [fichier.csv]      (défaut : raw_path de dataset.yaml)

Sortie : un résumé JSON sur la sortie standard et dans reports/validation.json.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.dataset_config import DatasetConfig, load_config

REPORT_JSON = Path("reports/validation.json")


def run_checks(cfg: DatasetConfig, df: pd.DataFrame) -> dict[str, bool]:
    """Un booléen par contrôle (True = conforme au contrat)."""
    try:
        cfg.validate_columns(df)  # toutes les colonnes déclarées sont présentes
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return {"colonnes_du_contrat": False}  # inutile d'aller plus loin
    target = df[cfg.target]
    classes = set(target.dropna().astype(str))
    return {
        "colonnes_du_contrat": True,
        "table_non_vide": len(df) > 0,
        "cible_sans_valeur_manquante": bool(target.notna().all()),
        "cible_avec_classe_positive_et_autre": str(cfg.positive_class) in classes
        and len(classes) >= 2,
        "numeriques_de_type_nombre": all(
            pd.api.types.is_numeric_dtype(df[col]) for col in cfg.numeric
        ),
        "aucune_variable_entierement_vide": not any(df[col].isna().all() for col in cfg.features),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", help="fichier à valider (défaut : raw_path)")
    args = parser.parse_args()

    cfg = load_config()
    df = pd.read_csv(args.csv or cfg.raw_path)
    checks = run_checks(cfg, df)
    failed = [name for name, ok in checks.items() if not ok]
    has_target = cfg.target in df.columns and len(df) > 0
    summary = {
        "success": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": failed,
        "rows": len(df),
        "positive_rate": round(float(cfg.encode_target(df[cfg.target]).mean()), 4)
        if has_target
        else None,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
