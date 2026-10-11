# Mes notes du cours MLOps

## Reproduire le projet (Lab 1)
- Cloner le repo, puis `make up`, `make pf`, `uv sync` et `source .venv/bin/activate`.
- Identifiants du remote DVC (jamais commités) :
  `dvc remote modify --local storage access_key_id mlops` et
  `dvc remote modify --local storage secret_access_key mlops-lab-secret`.
- Même cluster (remote déjà rempli) : `dvc pull` puis `dvc repro` (tout est « skipping »).
- Autre cluster (remote vide) : `dvc repro` puis `dvc push` ; ne pas commiter le `dvc.lock`
  régénéré (seul le hash du modèle change).
