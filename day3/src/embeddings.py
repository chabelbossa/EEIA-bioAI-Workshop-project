# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
embeddings.py — chargeur pour les embeddings Evo2 supervisés (extraits une fois par
l'organisateur avec extract_evo2_embeddings.py).

Ce fichier est complet (code d'infrastructure/E-S, pas l'objectif pédagogique du jour) —
utilisez-le tel quel.
"""

from pathlib import Path

import numpy as np


def load_supervised_embeddings(embeddings_dir, split: str, layer: str = None):
    """Retourne (embeddings: (N, d), labels: (N,), ids: (N,)).

    layer : nom de la couche Evo2 à charger (ex. "blocks.26", "blocks.31").
            Par défaut, la couche principale enregistrée par l'extraction.
            Utilisez available_layers(...) pour voir ce que contient le fichier.
    """
    path = Path(embeddings_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lancez extract_evo2_embeddings.py d'abord."
        )
    data = np.load(path, allow_pickle=True)

    key = "embeddings" if layer is None else f"embeddings_{layer}"
    if key not in data.files:
        raise KeyError(
            f"couche {layer!r} absente de {path} — couches disponibles : "
            f"{available_layers(embeddings_dir, split)}"
        )
    return data[key], data["labels"], data["ids"]


def available_layers(embeddings_dir, split: str):
    """Liste les couches Evo2 présentes dans <split>.npz."""
    data = np.load(Path(embeddings_dir) / f"{split}.npz", allow_pickle=True)
    if "layers" in data.files:
        return [str(x) for x in data["layers"]]
    return [f.removeprefix("embeddings_") for f in data.files
            if f.startswith("embeddings_")]
