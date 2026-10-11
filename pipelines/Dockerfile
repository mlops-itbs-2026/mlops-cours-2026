# Fichier : pipelines/Dockerfile
# Image « projet » du pipeline : le code du projet (src/), le contrat dataset.yaml,
# params.yaml, la suite Great Expectations (gx/) et les tests comportementaux du modèle,
# avec leurs dépendances figées. TOUS les composants KFP (et les tasks Airflow) tournent
# dans cette image : le code est réutilisé tel quel, jamais recopié.
# Contexte de build = RACINE du repo (même tag que PIPELINE_IMAGE de components.py) :
#   docker build -f pipelines/Dockerfile --build-arg VERSION=<tag> \
#       -t localhost:5001/scoring-pipeline:<tag> .
FROM python:3.12-slim

WORKDIR /app

# 1. Les dépendances d'abord : cette couche, la plus lourde, reste en cache tant que
#    requirements.txt ne change pas (reconstruire après une modification de src/ = 2 s).
#    kfp sans ses dépendances suffit à exécuter un composant : c'est ce que KFP
#    installerait au démarrage de CHAQUE pod (install_kfp_package=False dans components.py).
COPY pipelines/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt \
    && pip install --no-cache-dir --root-user-action=ignore --no-deps kfp==2.17.0

# 2. Le code et le contrat, avec la même arborescence que le repo
COPY dataset.yaml params.yaml ./
COPY src/ src/
COPY gx/ gx/
COPY tests/ tests/
COPY pipelines/__init__.py pipelines/stage.py pipelines/

# src et pipelines importables depuis n'importe quel dossier, dataset.yaml trouvé partout.
# Pas d'utilisateur non-root ici : le lanceur KFP écrit les artefacts sous /minio, /s3…
# à la racine du conteneur (le Lab 4 durcira l'image de serving).
# DO_NOT_TRACK : ni MLflow ni Evidently n'envoient de télémétrie (GX : gx/*.yml).
ENV PYTHONPATH=/app \
    DATASET_CONFIG=/app/dataset.yaml \
    PYTHONUNBUFFERED=1 \
    DO_NOT_TRACK=true

ARG VERSION=dev
LABEL org.opencontainers.image.title="scoring-pipeline" \
      org.opencontainers.image.version="${VERSION}"
