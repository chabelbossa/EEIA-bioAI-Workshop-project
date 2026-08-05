# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
classifier_heads.py — petites têtes entraînées sur des embeddings gelés (embeddings Evo2
pour le teacher, ou latents SAE pour la piste bonus).
"""

import torch.nn as nn


class MLPHead(nn.Module):
    """MLP à une seule couche cachée -> logit binaire. Volontairement petit : l'embedding
    a déjà fait le gros du travail, cette tête n'a qu'à apprendre la frontière de décision."""

    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):  # x: (B, d_in)
        return self.net(x).squeeze(-1)  # (B,) logits
