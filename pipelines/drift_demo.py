# Fichier : pipelines/drift_demo.py
"""Démonstration de drift (Lab 3) : une nouvelle clientèle comparée à celle d'entraînement.

ingest -> prepare -> simulate_drift (test déplacé) -> detect_drift (train vs test déplacé)
Le scénario est dans params.yaml (drift.simulation). Par défaut le drift est signalé sans
bloquer (fail_on_drift=False) : le run réussit et le rapport montre le drift.
Compilation : python -m pipelines.drift_demo   (écrit drift_demo.yaml)
"""

from pathlib import Path

import yaml
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Output

from pipelines.components import PIPELINE_IMAGE, detect_drift, ingest_data, prepare_features

DRIFT = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["drift"]


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def simulate_drift(data: Input[Dataset], drifted: Output[Dataset]) -> None:
    """Déplace les données selon le scénario de params.yaml (python -m src.simulate drift)."""
    from pipelines.stage import run_stage

    clean, shifted = "data/drift/clean.csv", "data/drift/drifted.csv"
    run_stage(
        "src.simulate",
        args=["drift", clean, shifted],
        inputs={clean: data.path},
        outputs={shifted: drifted.path},
    )


@dsl.pipeline(name="scoring-drift-demo", description="Drift simulé détecté par detect_drift")
def drift_pipeline(
    ks_alpha: float = DRIFT["ks_alpha"],
    psi_threshold: float = DRIFT["psi_threshold"],
    drift_share: float = DRIFT["drift_share"],
    fail_on_drift: bool = False,  # True : le drift fait échouer le run (gate)
):
    ingest_task = ingest_data()
    prep_task = prepare_features(raw_data=ingest_task.outputs["raw_data"])
    drifted_task = simulate_drift(data=prep_task.outputs["test_data"])
    drift_task = detect_drift(
        reference_data=prep_task.outputs["train_data"],
        current_data=drifted_task.outputs["drifted"],
        ks_alpha=ks_alpha,
        psi_threshold=psi_threshold,
        drift_share=drift_share,
        fail_on_drift=fail_on_drift,
    )
    drift_task.set_memory_limit("1Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(drift_pipeline, package_path="drift_demo.yaml")
