# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
viz.py — visualisation de l'activité des caractéristiques du SAE le long de la séquence.

**À COMPLÉTER** : `plot_sae_feature_activity` ci-dessous.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sae_feature_activity(feature_matrix: np.ndarray, feature_indices,
                               title="SAE feature activity along the sequence"):
    """feature_matrix: (seq_len, n_selected_features) — une ligne par caractéristique,
    axe x = position du nucléotide. Sert à repérer des motifs d'activation périodiques
    (par ex. période 3, cadre de lecture des codons).

    TODO :
      1. Créez une figure/axe (plt.subplots(figsize=(12, 3 + 0.5 * len(feature_indices)))).
      2. Pour i, feat_idx in enumerate(feature_indices) :
         tracez ax.plot(feature_matrix[:, i] + i * 1.2, label=f"feature {feat_idx}")
         (le décalage `+ i * 1.2` empile les courbes verticalement pour la lisibilité)
      3. Ajoutez xlabel ("Position in sequence (nt)"), retirez les yticks (ax.set_yticks([])),
         ajoutez le titre et une légende (loc="upper right", fontsize=8), puis affichez.
    """
    raise NotImplementedError("TODO : implémentez plot_sae_feature_activity")
