# Fichier : src/validate_model.py
"""Valide le modèle entraîné avant son enregistrement (gate du Lab 3).

    python -m src.validate_model      (models/model, data/processed/train.csv et test.csv)

1. Evidently (ClassificationPreset) sur le jeu de test, au seuil de décision du modèle,
   avec le jeu d'entraînement en référence et deux tests explicites :
   - la métrique primaire de dataset.yaml atteint promotion_threshold ;
   - la ROC AUC de test ne s'écarte pas de plus de 20 % de celle d'entraînement
     (sur-apprentissage).
2. Les tests comportementaux de tests/test_model_behavior.py (pytest).
Code retour 1 si un test échoue. Sorties : reports/model_validation.json, .md, .html.
"""

import json
from pathlib import Path
from xml.etree import ElementTree

import mlflow
import pandas as pd
import pytest
from mlflow.models import Model

from src.dataset_config import DatasetConfig, load_config
from src.train import MODEL_DIR, TEST_CSV, TRAIN_CSV, compute_metrics

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_TESTS = ROOT / "tests" / "test_model_behavior.py"
REPORT_JSON = Path("reports/model_validation.json")
REPORT_MD = Path("reports/model_validation.md")
REPORT_HTML = Path("reports/model_validation.html")
JUNIT_XML = Path("reports/model_behavior.xml")
MAX_TRAIN_TEST_GAP = 0.2  # écart relatif de ROC AUC toléré entre entraînement et test
# Métriques de dataset.yaml que le preset sait tester (nom de l'argument « <x>_tests »)
EVIDENTLY_NAMES = {
    "roc_auc": "rocauc",
    "f1": "f1score",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}
PRESET_METRICS = ["accuracy", "precision", "recall", "f1score", "rocauc", "logloss",
                  "tpr", "tnr", "fpr", "fnr"]  # fmt: skip


def evidently_checks(cfg: DatasetConfig, model, threshold: float) -> list:
    """Rapport Evidently (HTML) et résultat de ses tests."""
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset
    from evidently.tests import Reference, eq, gte

    definition = DataDefinition(
        classification=[BinaryClassification(target="target", prediction_probas="proba",
                                             pos_label=1)]
    )  # fmt: skip

    def dataset(path: Path) -> Dataset:
        X, y = cfg.read_xy(path)
        frame = pd.DataFrame({"target": y.to_numpy(), "proba": model.predict_proba(X)[:, 1]})
        return Dataset.from_pandas(frame, data_definition=definition)

    # Tests explicites seulement. Une liste vide désactive le test par défaut d'Evidently
    # (« mieux qu'un modèle naïf »), inadapté au seuil coût-optimal : accuracy, TNR et FPR
    # y sont volontairement moins bonnes que celles d'un modèle qui accorde tout.
    tests = {f"{name}_tests": [] for name in PRESET_METRICS}
    tests["rocauc_tests"] = [eq(Reference(relative=MAX_TRAIN_TEST_GAP))]  # vs entraînement
    if cfg.primary_metric in EVIDENTLY_NAMES:
        name = EVIDENTLY_NAMES[cfg.primary_metric]
        tests[f"{name}_tests"].append(gte(cfg.promotion_threshold))

    report = Report([ClassificationPreset(probas_threshold=threshold, **tests)])
    snapshot = report.run(dataset(TEST_CSV), dataset(TRAIN_CSV))  # courant, référence
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_HTML))
    return [
        {"name": t["name"], "success": getattr(t["status"], "value", t["status"]) == "SUCCESS"}
        for t in snapshot.dict()["tests"]
    ]


def behavior_checks() -> list:
    """Lance pytest sur les tests comportementaux ; un résultat par test (rapport JUnit)."""
    JUNIT_XML.parent.mkdir(parents=True, exist_ok=True)
    quiet = ["-q", "-p", "no:cacheprovider", "-W", "ignore::pytest.PytestAssertRewriteWarning"]
    code = pytest.main([*quiet, str(BEHAVIOR_TESTS), f"--junitxml={JUNIT_XML}"])
    if code not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        return [{"name": f"pytest : code retour {int(code)}", "success": False}]
    return [
        {
            "name": case.get("name"),
            "success": case.find("failure") is None and case.find("error") is None,
            "skipped": case.find("skipped") is not None,
        }
        for case in ElementTree.parse(JUNIT_XML).getroot().iter("testcase")
    ]


def markdown(summary: dict) -> str:
    """Tableau des tests (affiché par l'UI Kubeflow)."""
    lines = [
        f"### Validation du modèle : {summary['n_failed']} échec(s) sur {summary['n_checks']}",
        "",
        "| Famille | Test | Statut |",
        "|---|---|---|",
    ]
    for family in ("evidently", "behavior"):
        for c in summary[family]:
            status = "ignoré" if c.get("skipped") else ("OK" if c["success"] else "ÉCHEC")
            lines.append(f"| {family} | {c['name']} | {status} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_config()
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    threshold = Model.load(str(MODEL_DIR)).metadata["decision_threshold"]
    X, y = cfg.read_xy(TEST_CSV)
    metrics = compute_metrics(cfg, y, model.predict_proba(X)[:, 1], threshold)

    evidently, behavior = evidently_checks(cfg, model, threshold), behavior_checks()
    failed = [c["name"] for c in [*evidently, *behavior] if not c["success"]]
    summary = {
        "success": not failed,
        "n_checks": len(evidently) + len(behavior),
        "n_failed": len(failed),
        "failed": failed,
        cfg.primary_metric: round(float(metrics[cfg.primary_metric]), 4),
        "decision_threshold": round(threshold, 4),
        "evidently": evidently,
        "behavior": behavior,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    REPORT_MD.write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    print("Modèle VALIDÉ" if not failed else f"Modèle REFUSÉ : {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
