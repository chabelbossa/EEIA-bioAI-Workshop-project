# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
featurize.py — transformez une chaîne de nucléotides brute en caractéristiques utilisables
par un modèle.

Deux encodages, correspondant aux deux modèles de référence du Jour 1 :
    kmer_frequencies(seq, k)   -> entrée pour le ML classique (vecteur de comptage de k-mers)
    one_hot(seq)               -> entrée pour le deep learning (CNN sur les canaux A/C/G/T)

**À COMPLÉTER** : les corps de `kmer_frequencies` et `one_hot` ci-dessous. Le reste
(vocabulaire, fonctions batch) est fourni.
"""

import itertools
from functools import lru_cache

import numpy as np

BASES = "ACGT"


@lru_cache(maxsize=None)
def _kmer_vocab(k: int):
    """Liste ordonnée des 4**k k-mers possibles (fourni)."""
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def kmer_frequencies(seq: str, k: int = 4) -> np.ndarray:
    """Vecteur de fréquences de k-mers, normalisé (vocabulaire fixe de 4**k k-mers)."""
    vocab = _kmer_vocab(k)
    index = {kmer: i for i, kmer in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        idx = index.get(kmer)
        if idx is not None:  # on ignore les k-mers contenant un N ou une base ambiguë
            counts[idx] += 1
            n += 1
    if n > 0:
        counts /= n
    return counts


def kmer_matrix(seqs, k: int = 4) -> np.ndarray:
    """Empile kmer_frequencies sur une liste de séquences (fourni)."""
    return np.stack([kmer_frequencies(s, k) for s in seqs])


def one_hot(seq: str, length: int = None) -> np.ndarray:
    """Encode une séquence en one-hot -> forme (4, length), canaux = A, C, G, T.

    Les bases ambiguës (N, etc.) donnent une colonne entièrement à zéro. Les séquences
    sont tronquées/complétées (avec des colonnes de zéros) à `length` si fourni.
    """
    length = length or len(seq)
    arr = np.zeros((4, length), dtype=np.float32)
    base_to_idx = {b: i for i, b in enumerate(BASES)}
    for i, base in enumerate(seq[:length]):
        idx = base_to_idx.get(base)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def one_hot_batch(seqs, length: int = None) -> np.ndarray:
    """Empile one_hot sur une liste de séquences (fourni)."""
    length = length or max(len(s) for s in seqs)
    return np.stack([one_hot(s, length) for s in seqs])
