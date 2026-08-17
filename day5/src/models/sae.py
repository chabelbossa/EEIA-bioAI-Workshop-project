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
        f = F.relu(x @ self.W + self.b_enc)
        values, indices = torch.topk(f, self.k, dim=-1)
        sparse_f = torch.zeros_like(f)
        return sparse_f.scatter(-1, indices, values)

    def decode(self, f):
        return f @ self.W.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def train_sae(sae, activations: torch.Tensor, epochs=10, lr=2e-4,
              batch_size=512, device="cpu", log_every=1):
    """activations: tenseur (N, d_in), déjà chargé/sous-échantillonné en mémoire — prévu
    pour quelques milliers à dizaines de milliers de vecteurs, pas le dump complet
    multi-Go.
    """
    history = {"recon_loss": [], "alive_frac": [], "explained_var": []}
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n_samples = activations.shape[0]
    total_var = activations.var().item()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        feature_active = torch.zeros(sae.W.shape[1], dtype=torch.bool, device=device)

        for i in range(0, n_samples, batch_size):
            batch_indices = perm[i:i + batch_size]
            x = activations[batch_indices].to(device)

            optimizer.zero_grad()
            x_hat, f = sae(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.shape[0]
            feature_active |= (f > 0).any(dim=0)

        mean_loss = epoch_loss / n_samples
        explained_var = 1.0 - (mean_loss / total_var) if total_var > 0 else 0.0
        alive_frac = feature_active.float().mean().item()

        history["recon_loss"].append(mean_loss)
        history["explained_var"].append(explained_var)
        history["alive_frac"].append(alive_frac)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {mean_loss:.6f} | "
                  f"Expl. Var: {explained_var:.4f} | Alive Features: {alive_frac * 100:.1f}%")

    return sae, history

