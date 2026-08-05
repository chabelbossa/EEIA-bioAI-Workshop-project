# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
data.py — charge les CSV de fenêtres étiquetées produits par build_dataset.py.

Ce fichier est complet (code d'infrastructure, pas l'objectif pédagogique du jour) —
utilisez-le tel quel.
"""

from pathlib import Path

import pandas as pd


def load_split(processed_dir, split: str) -> pd.DataFrame:
    """Charge un split (train/val/test) sous forme de DataFrame (colonnes : voir build_dataset.py)."""
    path = Path(processed_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lancez build_dataset.py d'abord pour le générer."
        )
    return pd.read_csv(path, dtype={"sequence": str})


def load_all(processed_dir):
    """Retourne {"train": df, "val": df, "test": df}."""
    return {split: load_split(processed_dir, split) for split in ("train", "val", "test")}


def subsample(df: pd.DataFrame, n: int, seed: int = 42, stratify_col: str = "label") -> pd.DataFrame:
    """Sous-échantillon équilibré par classe d'au plus n lignes au total (n // n_classes par classe)."""
    if len(df) <= n:
        return df
    per_class = n // df[stratify_col].nunique()
    return (
        df.groupby(stratify_col, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
