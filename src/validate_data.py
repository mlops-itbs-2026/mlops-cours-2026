# Fichier : src/validate_data.py
"""Valide un fichier de données contre le contrat Great Expectations du projet (gx/).

Gate du pipeline (Labs 2 et 3) : si une expectation échoue, le script sort avec le code 1
et l'orchestrateur (DVC, Kubeflow, Airflow, CI) s'arrête AVANT l'entraînement. Le Lab 3
remplace les contrôles simples du Lab 2 par une suite Great Expectations, sans changer la
façon de lancer le script :

    python -m src.validate_data [fichier.csv]     (défaut : raw_path de dataset.yaml)
    python -m src.validate_data --build-suite     ((ré)écrit la suite dans gx/)

La suite est GÉNÉRÉE depuis dataset.yaml et les données d'entraînement (domaines observés),
puis versionnée dans Git : elle ne change que si on la régénère volontairement.
Sorties : résumé JSON (sortie standard et reports/validation.json), rapport HTML autonome
(reports/validation.html) et Data Docs (gx/uncommitted/data_docs/local_site/index.html).
"""

import argparse
import html
import json
import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from src.dataset_config import DatasetConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # gx/ est à la racine du projet
SUITE = "scoring_contract"  # gx/expectations/scoring_contract.json
CHECKPOINT = "scoring_gate"  # gx/checkpoints/scoring_gate.json
REFERENCE_CSV = Path("data/processed/train.csv")  # données d'entraînement (src/prepare.py)
REPORT_JSON = Path("reports/validation.json")
REPORT_HTML = Path("reports/validation.html")
MOSTLY = 0.99  # tolérance des contrôles de valeurs : jusqu'à 1 % hors domaine
NUMERIC_TYPES = ["int64", "int32", "float64", "float32", "Int64", "Float64"]
TEXT_TYPES = ["object", "str", "string", "category"]


# --------------------------------------------------------------- construire la suite
def build_expectations(cfg: DatasetConfig, ref: pd.DataFrame) -> list:
    """Le contrat de données, déduit de dataset.yaml et des données de référence."""
    E = gx.expectations
    expectations = []
    add = expectations.append
    numeric = [c for c in cfg.columns if c != cfg.target and pd.api.types.is_numeric_dtype(ref[c])]
    categorical = [c for c in cfg.columns if c not in numeric and c != cfg.target]

    # 1. Structure : les colonnes du contrat sont présentes ; le lot n'est pas tronqué
    add(E.ExpectTableColumnsToMatchSet(column_set=cfg.columns, exact_match=False))
    add(E.ExpectTableRowCountToBeBetween(min_value=max(1, len(ref) // 10)))

    # 2. Chaque colonne : remplie comme à l'entraînement, et du bon type
    for col in cfg.columns:
        null_rate = float(ref[col].isna().mean())  # 0 : aucune valeur manquante tolérée
        mostly = {"mostly": round(max(0.0, 1 - 2 * null_rate), 3)} if null_rate else {}
        add(E.ExpectColumnValuesToNotBeNull(column=col, **mostly))
        types = NUMERIC_TYPES if pd.api.types.is_numeric_dtype(ref[col]) else TEXT_TYPES
        add(E.ExpectColumnValuesToBeInTypeList(column=col, type_list=types))

    # 3. Numériques : bornes larges (même signe, au plus deux fois l'étendue observée).
    #    Elles attrapent les erreurs grossières (signe, unité) ; les variations naturelles
    #    relèvent de la détection de drift.
    for col in numeric:
        lo, hi = float(ref[col].min()), float(ref[col].max())
        span = hi - lo
        add(
            E.ExpectColumnValuesToBeBetween(
                column=col,
                min_value=0.0 if lo >= 0 else lo - span,
                max_value=hi + span,
                mostly=MOSTLY,
            )
        )

    # 4. Catégorielles : modalités vues à l'entraînement (une autre est inconnue du modèle)
    for col in categorical:
        observed = sorted(ref[col].dropna().unique().tolist(), key=str)
        add(E.ExpectColumnValuesToBeInSet(column=col, value_set=observed, mostly=MOSTLY))

    # 5. Cible : classes connues ; part de la classe positive entre la moitié et le double
    #    de celle de l'entraînement (« mostly » = part minimale des valeurs dans l'ensemble)
    classes = sorted(ref[cfg.target].dropna().unique().tolist(), key=str)
    positive = [c for c in classes if str(c) == str(cfg.positive_class)]
    others = [c for c in classes if str(c) != str(cfg.positive_class)]
    rate = float(cfg.encode_target(ref[cfg.target]).mean())
    add(E.ExpectColumnValuesToBeInSet(column=cfg.target, value_set=classes))
    add(
        E.ExpectColumnValuesToBeInSet(
            column=cfg.target,
            value_set=positive,
            mostly=round(rate / 2, 3),
            description=f"Au moins {rate / 2:.0%} de « {cfg.positive_class} »",
        )
    )
    add(
        E.ExpectColumnValuesToBeInSet(
            column=cfg.target,
            value_set=others,
            mostly=round(max(0.0, 1 - 2 * rate), 3),
            description=f"Au plus {min(1.0, 2 * rate):.0%} de « {cfg.positive_class} »",
        )
    )
    return expectations


def save_suite(expectations: list) -> gx.ExpectationSuite:
    """Écrit dans gx/ la suite, ses données (pandas), la validation et le checkpoint."""
    context = gx.get_context(mode="file", project_root_dir=PROJECT_ROOT)  # crée gx/
    context.enable_analytics(False)  # aucune télémétrie envoyée à GX
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    context.variables.save()
    # Source « pandas » : le DataFrame est fourni à chaque exécution (batch_parameters)
    batch = (
        context.data_sources.add_or_update_pandas("scoring")
        .add_dataframe_asset("dataframe")
        .add_batch_definition_whole_dataframe("full")
    )
    suite = context.suites.add_or_update(gx.ExpectationSuite(name=SUITE, expectations=expectations))
    definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=SUITE, data=batch, suite=suite)
    )
    context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=CHECKPOINT,
            validation_definitions=[definition],
            actions=[gx.checkpoint.UpdateDataDocsAction(name="data_docs")],
            result_format={"result_format": "SUMMARY"},
        )
    )
    return suite


# --------------------------------------------------------------------- valider
def validate(cfg: DatasetConfig, df: pd.DataFrame):
    """Lance le checkpoint versionné sur df ; renvoie le résultat de la validation."""
    if not (PROJECT_ROOT / "gx" / "checkpoints" / f"{CHECKPOINT}.json").exists():
        raise SystemExit("Suite absente : lancez python -m src.validate_data --build-suite")
    missing = [c for c in cfg.columns if c not in df.columns]
    if missing:
        print(f"Colonnes déclarées dans dataset.yaml absentes des données : {missing}",
              file=sys.stderr)  # fmt: skip
    context = gx.get_context(mode="file", project_root_dir=PROJECT_ROOT)
    result = context.checkpoints.get(CHECKPOINT).run(batch_parameters={"dataframe": df})
    return next(iter(result.run_results.values()))


def check_name(r) -> str:
    """Nom lisible d'une expectation : type(colonne), et sa description s'il y en a une."""
    config = r.expectation_config
    name = config.type + (f"({config.kwargs['column']})" if "column" in config.kwargs else "")
    return f"{name} : {config.description}" if config.description else name


def run_checks(cfg: DatasetConfig, df: pd.DataFrame) -> dict[str, bool]:
    """Un booléen par expectation de la suite (True = conforme au contrat)."""
    return {check_name(r): bool(r.success) for r in validate(cfg, df).results}


def expected(r) -> str:
    """Les paramètres de l'expectation : bornes, ensemble de valeurs, tolérance…"""
    kwargs = r.expectation_config.kwargs
    return ", ".join(
        f"{k}={str(v)[:80]}" for k, v in kwargs.items() if k not in ("column", "batch_id")
    )


def observed(r) -> str:
    """Ce que GX a observé : valeur, ou nombre et exemples de valeurs hors contrat."""
    if not r.success and r.exception_info.get("raised_exception"):
        return "erreur : " + str(r.exception_info.get("exception_message"))[:150]
    res = r.result or {}
    if "unexpected_count" in res:
        sample = ", ".join(map(str, (res.get("partial_unexpected_list") or [])[:5]))
        text = f"{res['unexpected_count']} hors contrat ({res['unexpected_percent']:.1f} %)"
        return f"{text} : {sample}…" if sample and not r.success else text
    return str(res.get("observed_value", ""))


def write_html(results, summary: dict, path: Path) -> None:
    """Rapport HTML autonome (un seul fichier) : affiché par l'UI Kubeflow."""
    rows = "".join(
        f"<tr class='{'ok' if r.success else 'ko'}'><td>{html.escape(check_name(r))}</td>"
        f"<td>{html.escape(expected(r))}</td><td>{html.escape(observed(r))}</td>"
        f"<td>{'OK' if r.success else 'ÉCHEC'}</td></tr>\n"
        for r in sorted(results, key=lambda r: r.success)  # les échecs en premier
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><title>Validation des données</title>
<style>body{{font-family:sans-serif;margin:2em}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
.ok td:last-child{{color:#1a7f37}} .ko td:last-child{{color:#cf222e;font-weight:bold}}</style>
</head><body><h1>Suite {SUITE} : {summary["n_failed"]} échec(s) sur {summary["n_checks"]}</h1>
<p>{summary["rows"]} lignes, part de la classe positive : {summary["positive_rate"]}</p>
<table><tr><th>Expectation</th><th>Attendu</th><th>Observé</th><th>Statut</th></tr>
{rows}</table>
</body></html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", help="fichier à valider (défaut : raw_path)")
    parser.add_argument(
        "--build-suite",
        action="store_true",
        help=f"(ré)écrit la suite depuis dataset.yaml et csv (défaut : {REFERENCE_CSV})",
    )
    args = parser.parse_args()
    cfg = load_config()

    if args.build_suite:
        suite = save_suite(build_expectations(cfg, pd.read_csv(args.csv or REFERENCE_CSV)))
        print(f"Suite {SUITE} : {len(suite.expectations)} expectations -> gx/expectations/")
        return 0

    df = pd.read_csv(args.csv or cfg.raw_path)
    results = validate(cfg, df).results
    failed = [check_name(r) for r in results if not r.success]
    has_target = cfg.target in df.columns and len(df) > 0
    summary = {
        "success": not failed,
        "n_checks": len(results),
        "n_failed": len(failed),
        "failed": failed,
        "rows": len(df),
        "positive_rate": round(float(cfg.encode_target(df[cfg.target]).mean()), 4)
        if has_target
        else None,
        "suite": SUITE,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_html(results, summary, REPORT_HTML)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
