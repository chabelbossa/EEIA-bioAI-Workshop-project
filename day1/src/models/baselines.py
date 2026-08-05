# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
baselines.py — modèles de référence du Jour 1 : k-mer + ML classique, et one-hot + CNN.

**À COMPLÉTER** : `make_kmer_classifier` et la classe `OneHotCNN` ci-dessous.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def make_kmer_classifier(kind: str = "logreg"):
    """kind: "logreg" ou "rf" — entraîné directement avec .fit(X, y) de sklearn.

    TODO : retournez
      - une `LogisticRegression(max_iter=1000)` si kind == "logreg"
      - une `RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)`
        si kind == "rf"
      - levez une `ValueError` pour toute autre valeur de `kind`
    """
    raise NotImplementedError("TODO : implémentez make_kmer_classifier")


class OneHotCNN(nn.Module):
    """Petit CNN 1D sur des fenêtres one-hot (4, L) de nucléotides -> logit binaire."""

    def __init__(self, seq_len: int, channels: int = 32):
        super().__init__()
        # TODO : construisez self.net, un nn.Sequential avec :
        #   Conv1d(4, channels, kernel_size=9, padding=4)
        #   ReLU()
        #   MaxPool1d(2)
        #   Conv1d(channels, channels, kernel_size=9, padding=4)
        #   ReLU()
        #   AdaptiveAvgPool1d(1)
        # puis self.head = nn.Linear(channels, 1)
        raise NotImplementedError("TODO : construisez l'architecture du CNN")

    def forward(self, x):  # x: (B, 4, L)
        # TODO :
        #   1. feats = self.net(x).squeeze(-1)   # (B, channels)
        #   2. return self.head(feats).squeeze(-1)  # (B,) logits
        raise NotImplementedError("TODO : implémentez le forward")
