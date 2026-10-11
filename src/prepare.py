# Fichier : src/prepare.py
"""Prépare les données : identifiant de demande, split stratifié, source du feature store.

Entrées : le fichier brut (raw_path de dataset.yaml) et params.yaml (section prepare).
Sorties : data/processed/train.csv, data/processed/test.csv (colonnes brutes + identifiant)
          et data/processed/features.parquet (source « offline » de Feast).
Aucun nom de colonne en dur : tout vient de dataset.yaml (load_config) et de params.yaml.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.dataset_config import load_config

PROCESSED = Path("data/processed")


def main() -> None:
    cfg = load_config()
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["prepare"]
    id_col = params["id_column"]

    df = pd.read_csv(cfg.raw_path)
    cfg.validate_columns(df)

    # Identifiant de demande (clé de l'entité Feast). S'il est absent du fichier brut,
    # on numérote les lignes : l'ordre du fichier est figé par DVC, donc l'identifiant
    # est reproductible.
    if id_col not in df.columns:
        df.insert(0, id_col, range(1, len(df) + 1))
    df = df[[id_col, *cfg.columns]]

    # Split stratifié sur la cible : même proportion de la classe positive des deux côtés
    y = cfg.encode_target(df[cfg.target])
    train, test = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)

    # Source offline de Feast : les features seules (jamais la cible), une ligne par
    # demande, avec un horodatage fixe (params.yaml) pour que le fichier soit reproductible.
    features = df[[id_col, *cfg.features]].copy()
    features[cfg.numeric] = features[cfg.numeric].astype("float64")
    features["event_timestamp"] = pd.Timestamp(params["event_timestamp"])
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    rate = cfg.encode_target(train[cfg.target]).mean(), cfg.encode_target(test[cfg.target]).mean()
    print(
        f"Train : {len(train)} / Test : {len(test)} · "
        f"classe '{cfg.positive_class}' : {rate[0]:.1%} / {rate[1]:.1%}"
    )


if __name__ == "__main__":
    main()
