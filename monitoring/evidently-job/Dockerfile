# Fichier : monitoring/evidently-job/Dockerfile
# Image du job de drift (CronJob). Contexte de build = RACINE du repo, comme l'API :
#   docker build -f monitoring/evidently-job/Dockerfile -t localhost:5001/scoring-drift:1.0.0 .
# Elle embarque dataset.yaml et src/dataset_config.py : la liste des variables vient du contrat.

FROM python:3.12-slim AS builder
WORKDIR /build
COPY monitoring/evidently-job/requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

FROM python:3.12-slim AS runtime
RUN groupadd --gid 10001 ml \
    && useradd --uid 10001 --gid ml --no-create-home --shell /usr/sbin/nologin ml
WORKDIR /app
COPY --from=builder /deps /deps
# DO_NOT_TRACK : désactive la télémétrie d'Evidently ; HOME : caches dans /tmp (non-root)
ENV PYTHONPATH=/deps \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DO_NOT_TRACK=1 \
    HOME=/tmp
COPY dataset.yaml ./
COPY src/__init__.py src/dataset_config.py src/
COPY monitoring/evidently-job/calc_drift.py monitoring/evidently-job/
USER 10001:10001
CMD ["python", "monitoring/evidently-job/calc_drift.py"]
