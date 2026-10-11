# Rapport — Lab 4 : packaging du modèle

Binôme : corrigé enseignant · Dataset : `german-credit` · Modèle : `scoring-model` v1
(Pipeline `ColumnTransformer` + `LogisticRegression(C=0.1, class_weight="balanced")`, ROC AUC
0,81 sur le jeu de test, seuil de décision 0,318)

## 1. Comparatif des tailles

| Image | Taille | CVE HIGH + CRITICAL |
|---|---|---|
| `scoring-api:naive` | ≈ 1,4 Go (`python:3.12` complet + tout le repo) | des dizaines à des centaines |
| `scoring-api:optimized` | ≈ 330 Mo (`python:3.12-slim` + 206 Mo de dépendances) | de 0 à quelques dizaines |
| `scoring-api:bentoml` | ≈ 500 Mo (`python:3.12-slim` + BentoML, scikit-learn, SciPy) | ≈ optimized |

Ordres de grandeur du corrigé : les tailles et les nombres de CVE exacts dépendent de la date
du build (mises à jour des images de base et de la base de vulnérabilités Trivy).

## 2. Comparatif des latences

1 000 appels par backend après 10 appels de chauffe, une ligne de `data/processed/test.csv`
(et 100 lignes pour le batch), CPU x86_64, Python 3.12.

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline | 4.143 | 6.015 | 6.734 |
| onnx_python | 0.069 | 0.092 | 0.098 |
| onnx_batch_100 | 0.320 | 0.462 | 0.558 |

## 3. Interprétation

- **Taille** : l'image naïve part de `python:3.12` (compilateurs, en-têtes, outils Debian :
  ≈ 1 Go) et copie tout le repo (`.git`, `data/`, et même le `.env` !). L'image optimisée
  part de `python:3.12-slim`, n'installe que les 7 dépendances de serving dans un stage
  séparé (pas de scikit-learn, SciPy ni MLflow : le graphe ONNX suffit) et ne copie que
  8 fichiers grâce à la liste blanche du `.dockerignore`.
- **Latence** : ONNX Runtime exécute un graphe statique compilé en C++. Le pipeline
  scikit-learn, lui, valide le DataFrame, sélectionne les colonnes par leur nom et enchaîne
  des appels Python par transformateur : ce surcoût fixe (≈ 4 ms) domine pour une seule
  ligne. ONNX est ≈ 60× plus rapide en unitaire ; en batch de 100 lignes, le coût par ligne
  tombe à ≈ 3 µs.
- **Scans** : l'image slim a beaucoup moins de paquets Debian, donc moins de CVE. ModelScan
  détecte l'opérateur `posix.system` du pickle piégé (CRITICAL) ; il ne sait pas analyser
  l'ONNX ni le skops (fichiers « skipped ») : leur sûreté vient du format (pas de code
  exécutable), pas du scan.

## 4. Choix recommandé

`scoring-api:optimized` : image la plus petite, la plus rapide, non-root, sans pickle, avec un
schéma d'entrée strict généré depuis `dataset.yaml`. Le modèle y est figé (une version du
Registry = un tag d'image), ce qui rend le déploiement et le rollback du Lab 5 triviaux.
BentoML reste intéressant pour prototyper vite (batching adaptatif, OpenAPI), au prix d'une
image plus lourde et d'un modèle sérialisé en pickle.

## 5. Pistes d'amélioration

- Quantization dynamique INT8 (`onnxruntime.quantization`) — peu utile ici (modèle de 11 ko),
  décisive pour les réseaux de neurones.
- Image distroless (`gcr.io/distroless/python3`) : ni shell ni gestionnaire de paquets (Lab 8).
- Supprimer `pip` et `setuptools` de l'image finale, épingler l'image de base par digest.
- Signer l'image (Cosign) et générer un SBOM (`trivy image --format cyclonedx`).
- Plusieurs workers uvicorn (ou `gunicorn -k uvicorn.workers.UvicornWorker`) selon les CPU.
