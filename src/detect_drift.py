# Fichier : src/detect_drift.py
"""Détecte un drift entre des données de référence et des données courantes (Lab 3).

    python -m src.detect_drift [reference.csv courant.csv] [--no-fail]
    (défaut : data/processed/train.csv et data/processed/test.csv)

Pour chaque colonne de dataset.yaml (features et cible) :
- significativité : p-value du test de Kolmogorov-Smirnov (numérique) ou du chi2 (catégorielle) ;
- magnitude : PSI (Population Stability Index) et, pour une numérique, distance de
  Wasserstein normalisée (en écarts-types de la référence).
Une colonne dérive si p-value < ks_alpha ET PSI >= psi_threshold : significatif ET important.
Gate : le lot dérive si la part de colonnes en drift atteint drift_share ; code retour 1
(sauf --no-fail). Seuils par défaut : section drift de params.yaml.
Sorties : reports/drift.json, reports/drift.md (tableau) et reports/drift.html (Evidently).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from src.dataset_config import DatasetConfig, load_config

REFERENCE_CSV = Path("data/processed/train.csv")
CURRENT_CSV = Path("data/processed/test.csv")
REPORT_JSON = Path("reports/drift.json")
REPORT_MD = Path("reports/drift.md")
REPORT_HTML = Path("reports/drift.html")
BINS = 10  # PSI : déciles de la référence


def psi(ref: pd.Series, cur: pd.Series) -> float:
    """Population Stability Index : < 0,1 stable ; 0,1-0,25 modéré ; > 0,25 drift."""
    ref, cur = ref.dropna(), cur.dropna()
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > BINS:
        # Intervalles = déciles de la référence (10 % de la référence dans chacun)
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    else:  # catégorielle, ou numérique à peu de valeurs : une case par modalité
        values = sorted(set(ref) | set(cur), key=str)
        r = ref.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
        c = cur.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)  # éviter log(0)
    return float(np.sum((c - r) * np.log(c / r)))


def column_drift(ref: pd.Series, cur: pd.Series, numeric: bool) -> dict:
    """Test de significativité et mesures de magnitude pour une colonne."""
    if numeric:
        p_value = ks_2samp(ref.dropna(), cur.dropna()).pvalue
        scale = ref.std() or 1.0
        wasserstein = wasserstein_distance(ref.dropna(), cur.dropna()) / scale
        test = "KS"
    else:  # table de contingence modalités x (référence, courant)
        sample = np.repeat(["reference", "courant"], [len(ref), len(cur)])
        table = pd.crosstab(pd.concat([ref, cur]).astype(str).to_numpy(), sample)
        p_value = chi2_contingency(table)[1] if len(table) > 1 else 1.0
        wasserstein, test = None, "chi2"
    return {
        "test": test,
        "p_value": float(p_value),
        "psi": psi(ref, cur),
        "wasserstein": None if wasserstein is None else float(wasserstein),
    }


def numeric_columns(cfg: DatasetConfig, ref: pd.DataFrame) -> list:
    """Colonnes traitées comme numériques : numériques du contrat (et sensibles numériques)."""
    return [
        c for c in cfg.columns
        if c not in (cfg.target, *cfg.categorical) and pd.api.types.is_numeric_dtype(ref[c])
    ]  # fmt: skip


def detect(cfg: DatasetConfig, ref, cur, ks_alpha: float, psi_threshold: float) -> list:
    """Une ligne par colonne du contrat : statistiques et verdict."""
    numeric = numeric_columns(cfg, ref)
    rows = []
    for col in cfg.columns:
        stats = column_drift(ref[col], cur[col], col in numeric)
        drift = stats["p_value"] < ks_alpha and stats["psi"] >= psi_threshold
        rows.append({"column": col, **stats, "drift": drift})
    return rows


def evidently_report(cfg: DatasetConfig, ref, cur, drift_share: float, path: Path) -> dict:
    """Rapport HTML Evidently (DataDriftPreset) ; renvoie son propre verdict pour comparer."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    numeric = numeric_columns(cfg, ref)
    definition = DataDefinition(
        numerical_columns=numeric,
        categorical_columns=[c for c in cfg.columns if c not in numeric],
    )
    report = Report([DataDriftPreset(columns=cfg.columns, drift_share=drift_share)])
    snapshot = report.run(
        Dataset.from_pandas(cur[cfg.columns], data_definition=definition),
        Dataset.from_pandas(ref[cfg.columns], data_definition=definition),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    count = snapshot.dict()["metrics"][0]["value"]  # DriftedColumnsCount
    return {"drifted_columns": int(count["count"]), "share": round(float(count["share"]), 4)}


def markdown(rows: list, summary: dict) -> str:
    """Tableau des statistiques par colonne (affiché par l'UI Kubeflow)."""
    lines = [
        f"### Drift : {summary['n_drifted']}/{summary['n_columns']} colonnes "
        f"({'DRIFT' if summary['drift_detected'] else 'pas de drift'})",
        "",
        "| Colonne | Test | p-value | PSI | Wasserstein | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["psi"]):
        w = "" if r["wasserstein"] is None else f"{r['wasserstein']:.3f}"
        lines.append(
            f"| {r['column']} | {r['test']} | {r['p_value']:.4f} | {r['psi']:.3f} | {w} "
            f"| {'DRIFT' if r['drift'] else 'OK'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    defaults = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["drift"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", nargs="?", default=str(REFERENCE_CSV))
    parser.add_argument("current", nargs="?", default=str(CURRENT_CSV))
    parser.add_argument("--ks-alpha", type=float, default=defaults["ks_alpha"])
    parser.add_argument("--psi-threshold", type=float, default=defaults["psi_threshold"])
    parser.add_argument("--drift-share", type=float, default=defaults["drift_share"])
    parser.add_argument("--no-fail", action="store_true", help="drift signalé, code retour 0")
    args = parser.parse_args()

    cfg = load_config()
    ref, cur = pd.read_csv(args.reference), pd.read_csv(args.current)
    rows = detect(cfg, ref, cur, args.ks_alpha, args.psi_threshold)
    drifted = [r["column"] for r in rows if r["drift"]]
    share = len(drifted) / len(rows)
    summary = {
        "drift_detected": share >= args.drift_share,
        "n_columns": len(rows),
        "n_drifted": len(drifted),
        "share_drifted": round(share, 4),
        "drifted": drifted,
        "thresholds": {
            "ks_alpha": args.ks_alpha,
            "psi_threshold": args.psi_threshold,
            "drift_share": args.drift_share,
        },
        "rows": {"reference": len(ref), "current": len(cur)},
        "evidently": evidently_report(cfg, ref, cur, args.drift_share, REPORT_HTML),
        "columns": rows,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    REPORT_MD.write_text(markdown(rows, summary), encoding="utf-8")
    print(markdown(rows, summary))
    print(f"Evidently (p-value seule) : {summary['evidently']['drifted_columns']} colonne(s)")
    verdict = "DRIFT DÉTECTÉ" if summary["drift_detected"] else "pas de drift"
    print(f"{verdict} : {drifted} ({share:.0%} des colonnes, seuil {args.drift_share:.0%})")
    return 1 if summary["drift_detected"] and not args.no_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
