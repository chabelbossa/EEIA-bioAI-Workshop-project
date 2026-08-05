# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
sae.py — autoencodeur parcimonieux Top-K minimal pour la piste bonus d'interprétabilité.
Entraîné sans aucune étiquette sur des activations Evo2 brutes (data/autoencoder/).

Volontairement minuscule : c'est un artefact pédagogique, pas la version à l'échelle
recherche.

**À COMPLÉTER** : `TopKSparseAutoencoder.encode`/`decode`/`forward`, et `train_sae`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAutoencoder(nn.Module):
    """Dictionnaire surcomplet (d_hidden > d_in) avec une contrainte stricte de parcimonie
    top-k : chaque vecteur d'activation est expliqué par exactement k « caractéristiques »."""

    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.k = k
        W = F.normalize(0.1 * torch.randn(d_in, d_hidden), dim=0)
        self.W = nn.Parameter(W)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        """TODO :
          1. f = ReLU(x @ self.W + self.b_enc)                      # (B, d_hidden)
          2. values, indices = torch.topk(f, self.k, dim=-1)        # garder les k plus
             grandes activations par exemple
          3. sparse_f = torch.zeros_like(f)
          4. return sparse_f.scatter_(-1, indices, values)          # remettre les valeurs
             top-k à leur place, tout le reste à 0
        """
        raise NotImplementedError("TODO : implémentez encode")

    def decode(self, f):
        """TODO : return f @ self.W.T + self.b_dec"""
        raise NotImplementedError("TODO : implémentez decode")

    def forward(self, x):
        """TODO : f = self.encode(x) ; return self.decode(f), f"""
        raise NotImplementedError("TODO : implémentez forward")


def train_sae(sae, activations: torch.Tensor, epochs=10, lr=2e-4,
              batch_size=512, device="cpu"):
    """activations: tenseur (N, d_in), déjà chargé/sous-échantillonné en mémoire — prévu
    pour quelques milliers à dizaines de milliers de vecteurs, pas le dump complet
    multi-Go.

    TODO :
      1. Placez `sae` sur `device`, créez un optimiseur Adam (lr=lr).
      2. Pour chaque époque, mélangez les indices, puis pour chaque mini-lot :
         - x = activations[indices du mini-lot].to(device)
         - remettez les gradients à zéro
         - x_hat, _ = sae(x)
         - loss = F.mse_loss(x_hat, x)
         - rétropropagez, faites un pas d'optimisation
         - accumulez epoch_loss += loss.item() * x.shape[0]
      3. Ajoutez la perte moyenne à history["recon_loss"], affichez-la
      4. Retournez (sae, history)
    """
    history = {"recon_loss": []}
    raise NotImplementedError("TODO : implémentez train_sae")
