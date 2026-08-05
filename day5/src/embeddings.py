# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
embeddings.py — chargeur pour le dump d'activations Evo2 non supervisé (piste bonus SAE).

Ce fichier est complet (code d'infrastructure/E-S, pas l'objectif pédagogique du jour) —
utilisez-le tel quel.
"""

import json
from pathlib import Path

import numpy as np


def load_autoencoder_activations(autoencoder_dir, split: str, dtype=np.float32):
    """Memmap en lecture seule sur le dump d'activations Evo2 non étiquetées, hors ligne.

    split: "train" ou "eval" (ce jeu de données utilise "eval", pas "val").
    Retourne un memmap (total_tokens, d_model) — découpez-le, ne le chargez pas en entier.
    """
    split_dir = Path(autoencoder_dir) / split
    meta_path, data_path = split_dir / "meta.json", split_dir / "acts.dat"
    if not meta_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"acts.dat/meta.json manquants dans {split_dir}")

    meta = json.loads(meta_path.read_text())
    return np.memmap(
        data_path, mode="r", dtype=dtype,
        shape=(meta["total_tokens"], meta["d_model"]),
    )
