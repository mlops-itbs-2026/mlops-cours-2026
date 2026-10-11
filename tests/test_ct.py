# Fichier : tests/test_ct.py
"""Tests du ré-entraînement continu : pipelines KFP compilables, détection Evidently,
décisions du webhook. Sans KFP ni MLflow : la soumission du run est remplacée par un faux."""

import pytest
import yaml
from fastapi.testclient import TestClient
from kfp import compiler


def test_pipelines_kfp_compilent(tmp_path):
    """Les deux pipelines (Lab 2-3 et Lab 6) se compilent avec le code du repo."""
    from pipelines.ct_pipeline import ct_pipeline
    from pipelines.scoring_pipeline import scoring_pipeline

    compiler.Compiler().compile(scoring_pipeline, str(tmp_path / "scoring.yaml"))
    compiler.Compiler().compile(ct_pipeline, str(tmp_path / "ct.yaml"))
    spec = next(yaml.safe_load_all((tmp_path / "ct.yaml").read_text()))  # 1er document
    tasks = spec["root"]["dag"]["tasks"]
    for name in ("ingest-ct-data", "validate-data", "train-model", "promotion-gate"):
        assert name in tasks
    assert tasks["train-model"]["cachingOptions"] == {}  # cache désactivé : toujours ré-entraîner
    assert "register-challenger" in tasks["promotion-gate"]["dependentTasks"]


def test_drift_evidently(contract, make_frame):
    from pipelines.trigger_ct import drift_summary, make_drifted_batch

    reference = make_frame(contract)
    same = drift_summary(reference, reference.sample(frac=1.0, random_state=1))
    assert same == {"drift_share": 0.0, "drifted_features": []}
    drifted, columns = make_drifted_batch(reference, factor=1.5)
    summary = drift_summary(reference, drifted)
    assert columns and set(summary["drifted_features"]) == set(columns)
    assert summary["drift_share"] == pytest.approx(len(columns) / len(contract.features), 0.01)


@pytest.fixture
def webhook(monkeypatch):
    from pipelines import trigger_ct

    monkeypatch.setenv("CT_WEBHOOK_TOKEN", "secret-de-test")
    # réglages par défaut, quel que soit l'environnement (.env, CI)
    monkeypatch.setattr(trigger_ct, "MIN_DRIFT_SHARE", 0.1)
    monkeypatch.setattr(trigger_ct, "COOLDOWN_SECONDS", 600)
    monkeypatch.setattr(trigger_ct, "submit", lambda event: "run-42")  # pas de KFP
    monkeypatch.setattr(trigger_ct, "follow", lambda run_id, event: None)  # pas de suivi
    monkeypatch.setattr(trigger_ct, "STATE", {"busy": False, "last_trigger": 0.0,
                                              "last_run_id": None, "history": []})  # fmt: skip
    return TestClient(trigger_ct.app), trigger_ct


AUTH = {"Authorization": "Bearer secret-de-test"}


def test_webhook_authentification(webhook, monkeypatch):
    client, _ = webhook
    event = {"drift_share": 0.5}
    assert client.post("/drift", json=event).status_code == 401
    bad = {"Authorization": "Bearer mauvais"}
    assert client.post("/drift", json=event, headers=bad).status_code == 401
    monkeypatch.delenv("CT_WEBHOOK_TOKEN")
    assert client.post("/drift", json=event, headers=AUTH).status_code == 503


def test_webhook_decisions(webhook, contract):
    client, trigger_ct = webhook
    assert client.post("/drift", json={"drift_share": 2}, headers=AUTH).status_code == 422
    wrong = {"drift_share": 0.5, "dataset": "autre-dataset"}
    assert client.post("/drift", json=wrong, headers=AUTH).status_code == 422
    r = client.post("/drift", json={"drift_share": 0.01}, headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "skipped"

    event = {"source": "evidently-drift", "dataset": contract.name, "drift_share": 0.43,
             "drifted_features": ["x"], "champ_inconnu": "ignoré"}  # fmt: skip
    r = client.post("/drift", json=event, headers=AUTH)
    assert r.status_code == 202 and r.json() == {
        "status": "triggered",
        "reason": "dérive sur 43% des variables (x)",
        "run_id": "run-42",
    }
    assert client.get("/health").json()["last_run_id"] == "run-42"
    # un seul ré-entraînement à la fois, puis délai minimal entre deux
    r = client.post("/drift", json=event, headers=AUTH)
    assert r.json()["status"] == "skipped" and "en cours" in r.json()["reason"]
    trigger_ct.STATE["busy"] = False
    r = client.post("/drift", json=event, headers=AUTH)
    assert r.json()["status"] == "skipped" and "dernier ré-entraînement" in r.json()["reason"]


def test_webhook_alertmanager(webhook):
    client, _ = webhook
    alert = {"status": "firing", "labels": {"alertname": "ScoringDataDrift", "ct_trigger": "true",
             "feature": "x"}, "annotations": {"drift_share": "0.4"}}  # fmt: skip
    resolved = alert | {"status": "resolved"}
    r = client.post("/alertmanager", json={"alerts": [resolved]}, headers=AUTH)
    assert r.json()["status"] == "skipped"
    r = client.post("/alertmanager", json={"alerts": [alert]}, headers=AUTH)
    assert r.status_code == 202 and r.json()["run_id"] == "run-42"
