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


DEFAULT_MAX_ROWS = 4000   # autant de fenêtres que pour les embeddings Evo2


def load_split(processed_dir, split: str, max_rows=DEFAULT_MAX_ROWS,
               seed: int = 42) -> pd.DataFrame:
    """Charge un split (train/val/test) sous forme de DataFrame.

    max_rows : nombre maximum de fenêtres, **échantillonnées de façon équilibrée**
               (autant de codant que de non-codant, graine fixe -> toujours les
               mêmes lignes). `None` charge tout le split.

               La valeur par défaut vaut le nombre de fenêtres pour lesquelles on
               a extrait les embeddings Evo2 : tous les modèles de la semaine
               voient ainsi la même quantité de données, et la comparaison du
               Jour 4 mesure la représentation, pas la taille du jeu de données.
    """
    path = Path(processed_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lancez build_dataset.py d'abord pour le générer."
        )
    df = pd.read_csv(path, dtype={"sequence": str})
    if max_rows is not None:
        df = subsample(df, max_rows, seed=seed)
    return df


def load_all(processed_dir, max_rows=DEFAULT_MAX_ROWS, seed: int = 42):
    """Retourne {"train": df, "val": df, "test": df}.

    max_rows=None pour tout charger (nécessaire au Jour 3, où l'on doit retrouver
    les fenêtres exactes notées par le teacher via leurs `id`).
    """
    return {split: load_split(processed_dir, split, max_rows=max_rows, seed=seed)
            for split in ("train", "val", "test")}


def subsample(df: pd.DataFrame, n: int, seed: int = 42, stratify_col: str = "label") -> pd.DataFrame:
    """Sous-échantillon équilibré par classe d'au plus n lignes au total (n // n_classes par classe)."""
    if len(df) <= n:
        return df
    per_class = n // df[stratify_col].nunique()
    return (
        df.groupby(stratify_col, group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
