# Fichier : monitoring/evidently-job/calc_drift.py
"""Job de drift : compare les demandes reçues par l'API aux données d'entraînement.

1. Référence : data/processed/train.csv VERSIONNÉ, lu dans le S3 du cours à l'adresse
   que DVC lui a donnée (REFERENCE_URI, voir l'étape 6), ou un fichier local.
2. Données courantes : les prédictions journalisées par l'API (une ligne JSON par demande),
   lues dans Loki sur les WINDOW_MINUTES dernières minutes (ou CURRENT_PATH, un CSV).
3. Evidently 0.7 (DataDriftPreset) calcule un score de drift par variable.
4. Les métriques sont poussées au Pushgateway (scrapé par Prometheus), le rapport HTML est
   déposé sur S3 et, au-delà d'un seuil, le webhook de ré-entraînement (Lab 6) est appelé.

Aucun nom de colonne ici : les variables viennent de dataset.yaml.
Configuration par variables d'environnement (valeurs par défaut = CronJob du cluster).
"""

import io
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import httpx
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

ROOT = Path(__file__).resolve().parents[2]  # racine du repo (/app dans l'image)
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

env = os.environ.get
MODEL_NAME = env("MODEL_NAME", "scoring-model")
REFERENCE_URI = env("REFERENCE_URI", str(ROOT / "data" / "processed" / "train.csv"))
LOKI_URL = env("LOKI_URL", "http://loki.monitoring:3100")
LOKI_QUERY = env(
    "LOKI_QUERY", '{namespace="scoring", container="kserve-container"} |= `"event":"prediction"`'
)
CURRENT_PATH = env("CURRENT_PATH", "")  # si renseigné : CSV local au lieu de Loki
WINDOW_MINUTES = int(env("WINDOW_MINUTES", "60"))
MIN_ROWS = int(env("MIN_ROWS", "500"))  # en dessous, le test n'est pas fiable : on attend
MAX_ROWS = int(env("MAX_ROWS", "5000"))  # les demandes les plus récentes de la fenêtre
# PSI : l'indicateur historique du crédit (< 0,1 stable, 0,1-0,25 à surveiller, > 0,25 dérive)
DRIFT_METHOD = env("DRIFT_METHOD", "psi")
DRIFT_THRESHOLD = float(env("DRIFT_THRESHOLD", "0.25"))
PUSHGATEWAY_URL = env("PUSHGATEWAY_URL", "prometheus-pushgateway.monitoring:9091")
PUSH_JOB = env("PUSH_JOB", "scoring-drift")
REPORTS_URI = env("REPORTS_URI", "s3://monitoring/drift-reports")
CT_WEBHOOK_URL = env("CT_WEBHOOK_URL", "")  # ex. http://localhost:8095/drift (Lab 6)
CT_WEBHOOK_TOKEN = env("CT_WEBHOOK_TOKEN", "")
CT_DRIFT_SHARE = float(env("CT_DRIFT_SHARE", "0.1"))  # part de variables en dérive

cfg = load_config(env("DATASET_CONFIG", str(ROOT / "dataset.yaml")))


def s3_client():
    # Endpoint et identifiants lus dans l'environnement : AWS_ENDPOINT_URL,
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (Secret s3-credentials dans le cluster).
    return boto3.client("s3")


def read_table(uri: str) -> pd.DataFrame:
    """CSV ou Parquet, local ou s3://bucket/clé (DVC stocke sans extension : CSV)."""
    if uri.startswith("s3://"):
        u = urlparse(uri)
        body = s3_client().get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
        data = io.BytesIO(body)
        return pd.read_parquet(data) if body[:4] == b"PAR1" else pd.read_csv(data)
    return pd.read_parquet(uri) if uri.endswith(".parquet") else pd.read_csv(uri)


def features_only(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in cfg.features if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonnes de dataset.yaml absentes : {missing}")
    out = df[cfg.features].copy()
    out[cfg.numeric] = out[cfg.numeric].apply(pd.to_numeric, errors="coerce")
    out[cfg.categorical] = out[cfg.categorical].astype("string")
    return out


def fetch_from_loki() -> pd.DataFrame:
    """Demandes journalisées par l'API : les MAX_ROWS plus récentes de la fenêtre.

    Loki renvoie au plus 5 000 lignes par requête (max_entries_limit_per_query) : on
    lit de la plus récente à la plus ancienne, page par page, en reculant la borne de fin.
    """
    end_ns = time.time_ns()
    start_ns = end_ns - WINDOW_MINUTES * 60 * 10**9
    rows: list[dict] = []
    with httpx.Client(base_url=LOKI_URL, timeout=30.0) as client:
        while len(rows) < MAX_ROWS:
            r = client.get(
                "/loki/api/v1/query_range",
                params={
                    "query": LOKI_QUERY,
                    "start": start_ns,
                    "end": end_ns,
                    "limit": 5000,
                    "direction": "backward",
                },
            )
            r.raise_for_status()
            entries = [v for s in r.json()["data"]["result"] for v in s["values"]]
            if not entries:
                break
            for _, line in entries:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # ligne qui n'est pas du JSON (log d'accès...)
                if event.get("event") == "prediction":
                    rows.append(event["features"])
            end_ns = min(int(ts) for ts, _ in entries)  # page suivante : plus ancienne
            if len(entries) < 5000:
                break
    return pd.DataFrame(rows[:MAX_ROWS], columns=cfg.features)


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame):
    """Score et verdict par variable (verdict = test Evidently, valable pour toute méthode)."""
    definition = DataDefinition(numerical_columns=cfg.numeric, categorical_columns=cfg.categorical)
    report = Report(
        [DataDriftPreset(method=DRIFT_METHOD, threshold=DRIFT_THRESHOLD)], include_tests=True
    )
    snapshot = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    result = snapshot.dict()
    scores = {
        m["config"]["column"]: float(m["value"])
        for m in result["metrics"]
        if m["config"]["type"].endswith(":ValueDrift")
    }
    drifted = {
        t["metric_config"]["params"]["column"]
        for t in result["tests"]
        if t["metric_config"]["params"]["type"].endswith(":ValueDrift") and t["status"] == "FAIL"
    }
    return scores, drifted, snapshot


def push_metrics(rows: int, scores: dict[str, float], drifted: set[str]) -> None:
    registry = CollectorRegistry()
    labels = {"model": MODEL_NAME}
    Gauge("ml_drift_current_rows", "Demandes analysées", ["model"], registry=registry).labels(
        **labels
    ).set(rows)
    if scores:
        score = Gauge(
            "ml_drift_score",
            f"Score de drift ({DRIFT_METHOD}) par variable",
            ["model", "feature"],
            registry=registry,
        )
        detected = Gauge(
            "ml_drift_detected",
            "1 si la variable dérive (seuil dépassé)",
            ["model", "feature"],
            registry=registry,
        )
        for feature, value in scores.items():
            score.labels(feature=feature, **labels).set(value)
            detected.labels(feature=feature, **labels).set(int(feature in drifted))
        Gauge(
            "ml_drift_share", "Part des variables en dérive", ["model"], registry=registry
        ).labels(**labels).set(len(drifted) / len(scores))
        Gauge("ml_drift_threshold", "Seuil de drift utilisé", ["model"], registry=registry).labels(
            **labels
        ).set(DRIFT_THRESHOLD)
    # PUT : remplace toutes les métriques du groupe (pas de reste d'une exécution précédente)
    push_to_gateway(PUSHGATEWAY_URL, job=PUSH_JOB, registry=registry)
    print(f"Métriques poussées vers {PUSHGATEWAY_URL} (job={PUSH_JOB})")


def save_report(snapshot, stamp: str) -> str | None:
    if not REPORTS_URI:
        return None
    name = f"drift-{stamp}.html"
    if REPORTS_URI.startswith("s3://"):
        u = urlparse(REPORTS_URI)
        s3 = s3_client()
        try:
            s3.head_bucket(Bucket=u.netloc)
        except s3.exceptions.ClientError:
            s3.create_bucket(Bucket=u.netloc)  # premier run : le bucket n'existe pas encore
        key = f"{u.path.strip('/')}/{name}".lstrip("/")
        s3.put_object(
            Bucket=u.netloc,
            Key=key,
            Body=snapshot.get_html_str(as_iframe=False),
            ContentType="text/html; charset=utf-8",
        )
        uri = f"s3://{u.netloc}/{key}"
    else:
        Path(REPORTS_URI).mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(Path(REPORTS_URI) / name))
        uri = str(Path(REPORTS_URI) / name)
    print(f"Rapport Evidently : {uri}")
    return uri


def trigger_ct(share: float, drifted: set[str], report_uri: str | None) -> None:
    """Appelle le webhook de ré-entraînement continu du Lab 6 (POST /drift)."""
    body = {
        "source": "evidently-drift",
        "model": MODEL_NAME,
        "dataset": cfg.name,
        "drift_share": round(share, 4),
        "drifted_features": sorted(drifted),
        "reference": REFERENCE_URI,
        "window_minutes": WINDOW_MINUTES,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "report_uri": report_uri,
    }
    headers = {"Authorization": f"Bearer {CT_WEBHOOK_TOKEN}"} if CT_WEBHOOK_TOKEN else {}
    r = httpx.post(CT_WEBHOOK_URL, json=body, headers=headers, timeout=30.0)
    print(f"Webhook CT {CT_WEBHOOK_URL} -> HTTP {r.status_code} {r.text[:300]}")
    r.raise_for_status()


def main() -> int:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    reference = features_only(read_table(REFERENCE_URI))
    if CURRENT_PATH:
        current = features_only(read_table(CURRENT_PATH)).tail(MAX_ROWS)  # les plus récentes
        source = CURRENT_PATH
    else:
        current = features_only(fetch_from_loki())
        source = f"Loki, {WINDOW_MINUTES} dernières minutes"
    print(f"Référence : {len(reference)} lignes ({REFERENCE_URI})")
    print(f"Courant   : {len(current)} demandes ({source})")

    if len(current) < MIN_ROWS:
        print(f"Moins de {MIN_ROWS} demandes : drift non calculé (relancez avec plus de trafic).")
        if PUSHGATEWAY_URL:
            push_metrics(len(current), {}, set())
        return 0

    scores, drifted, snapshot = compute_drift(reference, current)
    share = len(drifted) / len(scores)
    print(f"\n{'variable':28s} {DRIFT_METHOD:>8s}  dérive (seuil {DRIFT_THRESHOLD:g})")
    for feature, value in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{feature:28s} {value:8.3f}  {'OUI' if feature in drifted else '-'}")
    print(f"\n{len(drifted)}/{len(scores)} variables en dérive (part {share:.2f})")

    if PUSHGATEWAY_URL:
        push_metrics(len(current), scores, drifted)
    report_uri = save_report(snapshot, stamp)
    if CT_WEBHOOK_URL and share >= CT_DRIFT_SHARE:
        trigger_ct(share, drifted, report_uri)
    elif CT_WEBHOOK_URL:
        print(f"Part {share:.2f} < {CT_DRIFT_SHARE} : pas de ré-entraînement demandé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
