# Fichier : serving/locust_report.py
"""Résume les CSV de Locust en un tableau Markdown, à coller dans le rapport (étape 9).

Usage (racine du repo) :
    uv run python serving/locust_report.py                          # reports/m5/run_*
    uv run python serving/locust_report.py "reports/m5/compare_*"   # comparatif (étape 8)
Une ligne par fichier <préfixe>_stats.csv : requêtes de scoring (POST) seulement.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "reports/m5/run_*"


def run_name(csv: Path) -> str:
    return csv.name.removesuffix("_stats.csv")


def natural_key(csv: Path):
    """run_100 < run_500 < run_1000 : tri numérique quand le nom contient un nombre."""
    numbers = re.findall(r"\d+", run_name(csv))
    return (int(numbers[-1]) if numbers else 0, run_name(csv))


def summarize(csv: Path) -> dict:
    stats = pd.read_csv(csv)
    post = stats[stats["Type"] == "POST"]
    if post.empty:
        raise SystemExit(f"{csv} : aucune requête POST (le test a-t-il tourné ?)")
    row = post.iloc[0]
    total, failures = int(row["Request Count"]), int(row["Failure Count"])
    return {
        "Run": run_name(csv),
        "Requêtes": total,
        "RPS": f"{row['Requests/s']:.1f}",
        "P50 (ms)": f"{row['50%']:.0f}",
        "P95 (ms)": f"{row['95%']:.0f}",
        "P99 (ms)": f"{row['99%']:.0f}",
        "Erreurs (%)": f"{100 * failures / total:.2f}" if total else "n/a",
    }


files = sorted(ROOT.glob(f"{PATTERN}_stats.csv"), key=natural_key)
if not files:
    raise SystemExit(f"Aucun fichier {PATTERN}_stats.csv : lancez d'abord Locust avec --csv")
rows = [summarize(f) for f in files]
print("| " + " | ".join(rows[0]) + " |")
print("|" + "---|" * len(rows[0]))
for r in rows:
    print("| " + " | ".join(str(v) for v in r.values()) + " |")
