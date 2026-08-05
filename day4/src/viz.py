# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
viz.py — le graphique payoff du Jour 4 : précision vs. latence vs. taille.

**À COMPLÉTER** : `plot_efficiency_tradeoff` ci-dessous.
"""

import matplotlib.pyplot as plt


def plot_efficiency_tradeoff(results: dict):
    """results: {nom_modèle: {"accuracy": float, "params": int, "latency_ms": float}}

    Graphique à bulles : x=latence, y=précision, taille de bulle=nombre de paramètres.
    C'est le payoff du bilan du Jour 4 — tous les modèles construits pendant la semaine
    sur un seul graphique.

    TODO :
      1. Créez une figure/axe (plt.subplots(figsize=(7, 5))).
      2. Calculez max_params = le plus grand "params" parmi tous les modèles de `results`.
      3. Pour chaque (nom, r) dans results.items() :
         - taille = 200 + 1800 * (r["params"] / max_params)
         - ax.scatter(r["latency_ms"], r["accuracy"], s=taille, alpha=0.6, label=nom)
         - ax.annotate(nom, (r["latency_ms"], r["accuracy"]),
                       textcoords="offset points", xytext=(8, 4), fontsize=9)
      4. Ajoutez xlabel ("Inference latency (ms/sample)"), ylabel ("Accuracy"),
         titre ("Accuracy vs. Latency (bubble size = parameter count)"), une grille
         légère (ax.grid(True, alpha=0.3)), puis affichez la figure.
    """
    raise NotImplementedError("TODO : implémentez plot_efficiency_tradeoff")
