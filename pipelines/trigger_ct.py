# Fichier : pipelines/trigger_ct.py
"""Ré-entraînement continu (Lab 6) : dérive Evidently -> webhook -> pipeline KFP -> GitOps.

Depuis la racine du repo, .env chargé, `make pf` et port-forward de KFP (8080) actifs :
  simulate   fabrique un lot récent étiqueté, avec une dérive (data/drift/current.csv)
  detect     rapport Evidently 0.7 (référence = train.csv), rangé dans MLflow, puis POST
             de l'événement au webhook
  send       POST d'un événement déjà calculé (fichier JSON, ex. produit par le Lab 7)
  serve      le webhook : décide, lance ct-pipeline dans KFP, suit le run et, si la gate
             a promu un nouveau @champion, lance la mise en production (pipelines/gitops.py)
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.dataset_config import load_config

REFERENCE = Path("data/processed/train.csv")  # données d'entraînement du champion (Lab 1)
CURRENT = Path("data/drift/current.csv")  # lot « récent » simulé (non versionné)
REPORT_DIR = Path("reports/drift")
DRIFT_EXPERIMENT = "scoring-drift"  # expérience MLflow des rapports de dérive
MODEL_NAME = "scoring-model"
KFP_HOST = os.getenv("KFP_HOST", "http://localhost:8080")  # port-forward de ml-pipeline-ui
KFP_EXPERIMENT = "scoring-ct"
PORT = int(os.getenv("CT_WEBHOOK_PORT", "8095"))
WEBHOOK_URL = os.getenv("CT_WEBHOOK_URL", f"http://localhost:{PORT}/drift")
MIN_DRIFT_SHARE = float(os.getenv("CT_MIN_DRIFT_SHARE", "0.1"))  # >= 10 % des variables
COOLDOWN_SECONDS = int(os.getenv("CT_COOLDOWN_SECONDS", "600"))
AUTO_RELEASE = os.getenv("CT_AUTO_RELEASE", "1") == "1"  # mise en prod après promotion
AUTO_PROMOTE = os.getenv("CT_AUTO_PROMOTE", "1") == "1"  # 0 : canary laissé au Lab 7
LOCAL_CI = os.getenv("CT_LOCAL_CI", "0") == "1"  # 1 : build local au lieu de GitHub

cfg = load_config()


# --- 1. Un lot récent, avec une dérive ---------------------------------------------------
def make_drifted_batch(df: pd.DataFrame, factor: float = 1.5) -> tuple[pd.DataFrame, list[str]]:
    """Variables numériques continues (> 20 valeurs distinctes) multipliées par factor.

    Simule un changement de population (montants, durées… plus élevés). Les étiquettes
    restent celles d'origine : un lot récent dont on connaît déjà l'issue des crédits.
    """
    columns = [c for c in cfg.numeric if df[c].nunique() > 20]
    drifted = df.copy()
    drifted[columns] = drifted[columns].astype("float64") * factor
    return drifted, columns


# --- 2. Détection avec Evidently 0.7 -----------------------------------------------------
def drift_summary(reference: pd.DataFrame, current: pd.DataFrame, html: Path | None = None):
    """Rapport DataDriftPreset résumé en un événement : part et liste des variables en dérive."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=cfg.numeric, categorical_columns=cfg.categorical)
    snapshot = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current[cfg.features], data_definition=definition),
        reference_data=Dataset.from_pandas(reference[cfg.features], data_definition=definition),
    )
    if html:
        html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html))
    share, drifted = 0.0, []
    for metric in snapshot.dict()["metrics"]:
        conf = metric["config"]
        if conf["type"].endswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif conf["type"].endswith("ValueDrift"):
            value, threshold = float(metric["value"]), float(conf["threshold"])
            # p-value (K-S, chi²…) : dérive si < seuil ; distance (Wasserstein…) : si >= seuil
            is_drift = value < threshold if "p_value" in conf["method"] else value >= threshold
            if is_drift:
                drifted.append(conf["column"])
    return {"drift_share": round(share, 4), "drifted_features": drifted}


def log_drift_run(summary: dict, current: Path, html: Path) -> dict:
    """Range le rapport ET le lot dans MLflow : le pipeline CT y lira ses données."""
    import mlflow

    mlflow.set_experiment(DRIFT_EXPERIMENT)
    with mlflow.start_run(run_name=f"drift-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tag("dataset", cfg.name)
        mlflow.set_tag("drifted_features", ",".join(summary["drifted_features"]))
        mlflow.log_metric("drift_share", summary["drift_share"])
        mlflow.log_artifact(str(html))
        mlflow.log_artifact(str(current), artifact_path="batch")
    base = f"runs:/{run.info.run_id}"
    return {"data_uri": f"{base}/batch/{current.name}", "report_uri": f"{base}/{html.name}"}


def post(url: str, event: dict) -> dict:
    token = os.getenv("CT_WEBHOOK_TOKEN", "")
    req = urllib.request.Request(
        url,
        data=json.dumps(event).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:  # 401 (jeton), 422 (événement), 502 (KFP)…
        return {"http_status": err.code, **json.loads(err.read() or b"{}")}


# --- 3. Le webhook -------------------------------------------------------------------------
class DriftEvent(BaseModel):
    """Événement de dérive ; les champs inconnus sont ignorés (producteurs variés)."""

    model_config = ConfigDict(extra="ignore")
    source: str = "evidently"
    model: str = MODEL_NAME
    dataset: str | None = None
    drift_share: float = Field(ge=0.0, le=1.0)
    drifted_features: list[str] = []
    reference: str | None = None
    window_minutes: int | None = None
    timestamp: str | None = None
    data_uri: str = ""  # lot étiqueté à apprendre (vide : la source, src.download)
    report_uri: str | None = None


app = FastAPI(title="Déclencheur du ré-entraînement (Lab 6)")
STATE: dict = {"busy": False, "last_trigger": 0.0, "last_run_id": None, "history": []}
LOCK = threading.Lock()


def check_token(authorization: str | None) -> None:
    expected = os.getenv("CT_WEBHOOK_TOKEN")
    if not expected:
        raise HTTPException(503, "CT_WEBHOOK_TOKEN n'est pas défini côté serveur")
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(given, expected):
        raise HTTPException(401, "jeton absent ou invalide")


def decide(event: DriftEvent) -> tuple[bool, str]:
    if event.dataset not in (None, cfg.name):
        raise HTTPException(422, f"dataset « {event.dataset} » : ce service gère « {cfg.name} »")
    if event.model != MODEL_NAME:
        return False, f"modèle « {event.model} » non géré"
    if event.drift_share < MIN_DRIFT_SHARE:
        return False, f"drift_share {event.drift_share} < {MIN_DRIFT_SHARE}"
    if STATE["busy"]:
        return False, f"un ré-entraînement est déjà en cours ({STATE['last_run_id']})"
    elapsed = time.time() - STATE["last_trigger"]
    if elapsed < COOLDOWN_SECONDS:
        return False, f"dernier ré-entraînement il y a {elapsed:.0f} s (< {COOLDOWN_SECONDS} s)"
    features = ", ".join(event.drifted_features) or "non précisées"
    return True, f"dérive sur {event.drift_share:.0%} des variables ({features})"


def submit(event: DriftEvent) -> str:
    """Compile ct-pipeline avec le code du repo et crée un run KFP ; renvoie son id."""
    import kfp

    from pipelines.ct_pipeline import ct_pipeline

    run = kfp.Client(host=KFP_HOST).create_run_from_pipeline_func(
        ct_pipeline,
        arguments={
            "data_uri": event.data_uri,
            "trigger_source": event.source,
            "drift_share": event.drift_share,
        },
        run_name=f"ct-{event.source}-{time.strftime('%Y%m%d-%H%M%S')}",
        experiment_name=KFP_EXPERIMENT,
        enable_caching=False,
    )
    return run.run_id


def promoted_version(run_id: str) -> str | None:
    """La version @champion si c'est CE run KFP qui l'a produite (tag de register_challenger)."""
    from mlflow import MlflowClient

    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    return champion.version if champion.tags.get("kfp_run_id") == run_id else None


def follow(run_id: str, event: DriftEvent) -> None:
    """En arrière-plan : attend la fin du run, puis met en production si promotion."""
    import kfp

    from pipelines import gitops

    entry = {"run_id": run_id, "source": event.source, "state": "RUNNING"}
    STATE["history"].append(entry)
    try:
        run = kfp.Client(host=KFP_HOST).wait_for_run_completion(run_id, timeout=3600)
        entry["state"] = str(run.state)
        version = promoted_version(run_id) if entry["state"] == "SUCCEEDED" else None
        entry["promoted_version"] = version
        print(f"Run {run_id} : {entry['state']}, nouveau @champion : {version}", flush=True)
        if version and AUTO_RELEASE:
            note = f"(run KFP {run_id[:8]}, drift_share={event.drift_share})"
            gitops.release(LOCAL_CI, AUTO_PROMOTE, percent=10, timeout=1800, note=note)
            entry["released"] = True
    except BaseException as exc:  # SystemExit des attentes de gitops compris
        entry["error"] = repr(exc)
        print(f"Run {run_id} : échec du suivi ou de la mise en prod : {exc!r}", flush=True)
    finally:
        STATE["busy"] = False


def handle(event: DriftEvent) -> JSONResponse:
    with LOCK:  # deux alertes simultanées ne lancent qu'un seul run
        retrain, reason = decide(event)
        if not retrain:
            return JSONResponse({"status": "skipped", "reason": reason}, status_code=200)
        try:
            run_id = submit(event)
        except Exception as exc:  # KFP injoignable (port-forward arrêté ?), refus de l'API…
            body = {"status": "error", "reason": f"soumission à KFP ({KFP_HOST}) : {exc!r}"}
            return JSONResponse(body, status_code=502)
        STATE.update(busy=True, last_trigger=time.time(), last_run_id=run_id)
    threading.Thread(target=follow, args=(run_id, event), daemon=True).start()
    body = {"status": "triggered", "reason": reason, "run_id": run_id}
    return JSONResponse(body, status_code=202)


@app.get("/health")
def health():
    return {"status": "ok", "busy": STATE["busy"], "last_run_id": STATE["last_run_id"]}


@app.get("/runs")
def runs():
    return STATE["history"]


@app.post("/drift")
def drift(event: DriftEvent, authorization: str | None = Header(None)):
    check_token(authorization)
    return handle(event)


@app.post("/alertmanager")
def alertmanager(payload: dict, authorization: str | None = Header(None)):
    """Format webhook d'Alertmanager : alertes « firing » portant le label ct_trigger="true"."""
    check_token(authorization)
    alerts = [
        a
        for a in payload.get("alerts", [])
        if a.get("status") == "firing" and a.get("labels", {}).get("ct_trigger") == "true"
    ]
    if not alerts:
        return JSONResponse({"status": "skipped", "reason": "aucune alerte ct_trigger"}, 200)

    def value(alert: dict, key: str, default=None):
        return alert.get("annotations", {}).get(key) or alert.get("labels", {}).get(key) or default

    event = DriftEvent(
        source="alertmanager",
        drift_share=max(float(value(a, "drift_share", 1.0)) for a in alerts),
        drifted_features=sorted({value(a, "feature") for a in alerts} - {None}),
        data_uri=next((value(a, "data_uri") for a in alerts if value(a, "data_uri")), ""),
    )
    return handle(event)


# --- Ligne de commande -------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--factor", type=float, default=1.5)
    p_sim.add_argument("--shuffle-labels", action="store_true", help="étiquettes corrompues")
    p_det = sub.add_parser("detect")
    p_det.add_argument("--reference", type=Path, default=REFERENCE)
    p_det.add_argument("--current", type=Path, default=CURRENT)
    p_det.add_argument("--no-send", action="store_true", help="rapport seulement")
    p_send = sub.add_parser("send")
    p_send.add_argument("--event", type=Path, required=True, help="fichier JSON")
    sub.add_parser("serve")
    args = parser.parse_args(argv)

    if args.cmd == "simulate":
        drifted, columns = make_drifted_batch(pd.read_csv(cfg.raw_path), args.factor)
        if args.shuffle_labels:  # test de la gate : un lot aux étiquettes mélangées
            drifted[cfg.target] = drifted[cfg.target].sample(frac=1, random_state=0).to_numpy()
        CURRENT.parent.mkdir(parents=True, exist_ok=True)
        drifted.to_csv(CURRENT, index=False)
        print(f"{CURRENT} : {len(drifted)} demandes, {', '.join(columns)} x {args.factor:g}")
    elif args.cmd == "detect":
        html = REPORT_DIR / "drift_report.html"
        summary = drift_summary(pd.read_csv(args.reference), pd.read_csv(args.current), html)
        event = {"source": "evidently-drift", "model": MODEL_NAME, "dataset": cfg.name,
                 **summary, "reference": str(args.reference),
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}  # fmt: skip
        event |= log_drift_run(summary, args.current, html)
        (REPORT_DIR / "drift_event.json").write_text(json.dumps(event, indent=2) + "\n")
        print(json.dumps(event, indent=2, ensure_ascii=False))
        if not args.no_send:
            print("Webhook :", json.dumps(post(WEBHOOK_URL, event), ensure_ascii=False))
    elif args.cmd == "send":
        event = json.loads(args.event.read_text(encoding="utf-8"))
        print("Webhook :", json.dumps(post(WEBHOOK_URL, event), ensure_ascii=False))
    else:
        import uvicorn

        if not os.getenv("CT_WEBHOOK_TOKEN"):
            sys.exit("Définissez CT_WEBHOOK_TOKEN dans .env (voir l'étape 6.2)")
        uvicorn.run(app, host="127.0.0.1", port=PORT)  # le poste seulement, pas le réseau


if __name__ == "__main__":
    main()
