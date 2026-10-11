# Fichier : pipelines/test_bad_data.py
"""Pipeline de test du gate (Lab 3) : sur des données corrompues, il doit ÉCHOUER.

ingest -> corrupt_data -> validate_data (échec attendu) -> prepare_features (jamais lancé)
Compilation : python -m pipelines.test_bad_data   (écrit test_bad_data.yaml)
"""

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Output

from pipelines.components import (
    PIPELINE_IMAGE,
    ingest_data,
    prepare_features,
    validate_data,
)


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def corrupt_data(raw_data: Input[Dataset], corrupted: Output[Dataset]) -> None:
    """Injecte trois bugs « silencieux » (python -m src.simulate corrupt)."""
    from pipelines.stage import run_stage

    clean, bad = "data/raw/clean.csv", "data/raw/corrupted.csv"
    run_stage(
        "src.simulate",
        args=["corrupt", clean, bad],
        inputs={clean: raw_data.path},
        outputs={bad: corrupted.path},
    )


@dsl.pipeline(name="scoring-bad-data-test", description="Le gate doit refuser ces données")
def bad_data_pipeline():
    ingest_task = ingest_data()
    corrupt_task = corrupt_data(raw_data=ingest_task.outputs["raw_data"])
    validate_task = validate_data(raw_data=corrupt_task.outputs["corrupted"])
    prep_task = prepare_features(raw_data=corrupt_task.outputs["corrupted"])
    prep_task.after(validate_task)  # ne doit jamais s'exécuter


if __name__ == "__main__":
    compiler.Compiler().compile(bad_data_pipeline, package_path="test_bad_data.yaml")
