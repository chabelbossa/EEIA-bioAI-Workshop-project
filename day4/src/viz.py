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

    La taille de chaque bulle représente le nombre de paramètres entraînables.
    """
    if not results:
        raise ValueError("results ne doit pas être vide")

    fig, ax = plt.subplots(figsize=(7, 5))
    max_params = max(max(result["params"] for result in results.values()), 1)

    for name, result in results.items():
        bubble_size = 200 + 1800 * (result["params"] / max_params)
        ax.scatter(
            result["latency_ms"], result["accuracy"],
            s=bubble_size, alpha=0.6, label=name,
        )
        ax.annotate(
            name, (result["latency_ms"], result["accuracy"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    ax.set_xlabel("Inference latency (ms/sample)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Latency (bubble size = parameter count)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig, ax
