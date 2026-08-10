# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
viz.py — visualisation de l'espace des embeddings (PCA/UMAP).

**À COMPLÉTER** : `plot_embedding_space` ci-dessous.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embedding_space(embeddings, labels, method="pca", title="Embedding space",
                          ax=None):
    """Nuage de points 2D des embeddings, colorés par étiquette codant(1)/non-codant(0).

    TODO :
      1. Si method == "pca" : réduisez `embeddings` à 2 dimensions avec
         `PCA(n_components=2).fit_transform(embeddings)`.
         Si method == "umap" : `import umap` puis
         `umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)`.
         Sinon : levez une ValueError.
      2. Si `ax` n'est pas fourni, créez une figure/axe avec plt.subplots(figsize=(6, 5)).
      3. Pour label in (1, 0) avec un nom ("coding"/"non-coding") et une couleur,
         tracez un scatter des points de ce label (ax.scatter(coords[mask, 0],
         coords[mask, 1], s=6, alpha=0.5, label=name, color=color)).
      4. Ajoutez le titre et la légende, affichez la figure si vous l'avez créée
         vous-même, puis retournez `coords`.
    """
    if method == "pca":
        coords = PCA(n_components=2).fit_transform(embeddings)
    elif method == "umap":
        import umap
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods are 'pca' and 'umap'.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True

    labels = np.asarray(labels)
    categories = [
        (1, "coding", "#1f77b4"),
        (0, "non-coding", "#ff7f0e"),
    ]

    for val, name, color in categories:
        mask = (labels == val)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=6,
            alpha=0.5,
            label=name,
            color=color,
        )

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend()

    if created_fig:
        plt.show()

    return coords

