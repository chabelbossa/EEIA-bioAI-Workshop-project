# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
eval.py — métriques d'évaluation, déjà apprises au Jour 1. Fichier complet — utilisez-le
tel quel.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_logits(logits, labels, threshold=0.0):
    """logits: scores bruts (seuil sigmoïde 0.5 == seuil de logit 0.0)."""
    preds = (logits > threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency_torch(model, example_input, n_runs=100, device="cpu"):
    """Latence moyenne d'inférence par échantillon, en millisecondes."""
    import time
    model = model.to(device).eval()
    example_input = example_input.to(device)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(example_input)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(example_input)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0
