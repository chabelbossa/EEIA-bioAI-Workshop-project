# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
eval.py — les métriques payoff du Jour 4 : accuracy/F1, nombre de paramètres, latence
d'inférence. Fonctionne pour les modules torch et les estimateurs sklearn.

**À COMPLÉTER** : les cinq fonctions ci-dessous.
"""

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_logits(logits, labels, threshold=0.0):
    """logits: scores bruts (seuil sigmoïde 0.5 == seuil de logit 0.0).

    TODO : calculez `preds` en seuillant `logits` (> threshold -> 1, sinon 0),
    puis retournez {"accuracy": ..., "f1": ...} avec accuracy_score/f1_score.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    preds = (logits > threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
    }


def evaluate_sklearn(model, X, y):
    """TODO : appelez model.predict(X), puis retournez {"accuracy": ..., "f1": ...}."""
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
    }


def count_params(model: torch.nn.Module) -> int:
    """TODO : sommez model.parameters()[i].numel() pour tous les paramètres."""
    return sum(p.numel() for p in model.parameters())


def measure_latency_torch(model, example_input, n_runs=100, device="cpu"):
    """Latence moyenne d'inférence par échantillon, en millisecondes.

    TODO :
      1. Mettez le modèle sur `device` et en mode eval().
      2. Faites 5 passages "warmup" (sans les chronométrer) — le premier appel est
         souvent plus lent (allocation mémoire, etc.), on ne veut pas le compter.
      3. Chronométrez `n_runs` appels avec time.perf_counter(), sans gradient
         (torch.no_grad()).
      4. Retournez (temps écoulé / n_runs) * 1000.0
    """
    model = model.to(device).eval()
    example_input = example_input.to(device)

    with torch.no_grad():
        for _ in range(5):  # warmup
            _ = model(example_input)

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(example_input)
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0


def measure_latency_sklearn(model, X, n_runs=100):
    """Même idée que measure_latency_torch, mais pour un estimateur sklearn
    (model.predict(X) au lieu de model(x))."""
    for _ in range(5):  # warmup
        _ = model.predict(X)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(X)
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0
