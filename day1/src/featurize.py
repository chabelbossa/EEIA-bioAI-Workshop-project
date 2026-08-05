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
    """Vecteur de fréquences de k-mers, normalisé (vocabulaire fixe de 4**k k-mers).

    TODO :
      1. Récupérez le vocabulaire avec `_kmer_vocab(k)` et construisez un dict
         {kmer: index}.
      2. Créez un vecteur de comptes de la bonne taille (np.zeros).
      3. Parcourez toutes les sous-chaînes de longueur k de `seq` en glissant de 1 chaque fois,
         et incrémentez le compte correspondant si le k-mer est dans le vocabulaire
         (ignorez silencieusement les k-mers contenant un 'N' ou une base ambiguë —
         ils ne seront simplement pas dans le vocabulaire).
      4. Normalisez le vecteur par le nombre total de k-mers valides comptés (pas par
         len(seq) !), pour obtenir une distribution de fréquences.
    """
    raise NotImplementedError("TODO : implémentez kmer_frequencies")


def kmer_matrix(seqs, k: int = 4) -> np.ndarray:
    """Empile kmer_frequencies sur une liste de séquences (fourni)."""
    return np.stack([kmer_frequencies(s, k) for s in seqs])


def one_hot(seq: str, length: int = None) -> np.ndarray:
    """Encode une séquence en one-hot -> forme (4, length), canaux = A, C, G, T.

    TODO :
      1. Si `length` n'est pas donné, utilisez len(seq).
      2. Créez un tableau de zéros de forme (4, length).
      3. Pour chaque position i (jusqu'à `length`), si la base est A/C/G/T, mettez
         un 1 à la ligne correspondante, colonne i.
      4. Les bases ambiguës (N, etc.) ou les positions au-delà de la séquence restent
         à zéro (encodage "tout à zéro" — pas d'erreur à lever).
    """
    raise NotImplementedError("TODO : implémentez one_hot")


def one_hot_batch(seqs, length: int = None) -> np.ndarray:
    """Empile one_hot sur une liste de séquences (fourni)."""
    length = length or max(len(s) for s in seqs)
    return np.stack([one_hot(s, length) for s in seqs])
