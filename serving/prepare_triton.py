# Fichier : serving/prepare_triton.py
"""Étape 8.1 — dépôt de modèles Triton sur le S3 du cours, à partir du modèle ONNX du Lab 4.

Arborescence produite (format imposé par Triton) :
    s3://models/triton/<nom du modèle>/config.pbtxt
    s3://models/triton/<nom du modèle>/1/model.onnx      (1 = version du modèle)
Le config.pbtxt ne liste AUCUNE colonne : Triton déduit les entrées et les sorties du graphe
ONNX (« auto-complete »). On y règle le batching dynamique et on publie, en paramètres, le
seuil de décision et la classe positive lus dans metadata.json : le client en a besoin pour
décider, Triton ne renvoyant que des probabilités.

Usage (racine du repo, après `set -a && source .env && set +a` et `make pf`) :
    uv run python serving/prepare_triton.py
"""

import json
import os
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
BUCKET = os.getenv("MODELS_BUCKET", "models")
PREFIX = "triton"  # storageUri de l'InferenceService : s3://models/triton

meta = json.loads((ROOT / "packaging/model/metadata.json").read_text())
name = meta["model_name"]
config = f"""name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: 64
dynamic_batching {{
  max_queue_delay_microseconds: 2000
}}
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
parameters {{
  key: "decision_threshold"
  value: {{ string_value: "{meta["threshold"]}" }}
}}
parameters {{
  key: "positive_class"
  value: {{ string_value: "{meta["positive_class"]}" }}
}}
parameters {{
  key: "model_version"
  value: {{ string_value: "{meta["model_version"]}" }}
}}
"""

# Point d'accès et identifiants lus dans l'environnement (.env : AWS_ENDPOINT_URL, AWS_*)
s3 = boto3.client("s3")
if BUCKET not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
    s3.create_bucket(Bucket=BUCKET)
s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{name}/config.pbtxt", Body=config.encode())
s3.upload_file(str(ROOT / "packaging/model/model.onnx"), BUCKET, f"{PREFIX}/{name}/1/model.onnx")

print(config)
for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/")["Contents"]:
    print(f"s3://{BUCKET}/{obj['Key']}  ({obj['Size']} octets)")
