"""Mesures reproductibles de taille et de latence pour les students du Jour 3."""

from __future__ import annotations

import io
import time

import numpy as np
import torch


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_feature_extraction(featurizer, sequences, repeats: int = 5) -> float:
    """Retourne la médiane du coût de transformation, en ms par séquence.

    Le featurizer reçoit toute la liste à chaque répétition. Le temps du premier
    appel de chauffe est exclu afin de ne pas compter l'initialisation des vocabulaires.
    """
    sequences = list(sequences)
    if not sequences:
        raise ValueError("sequences ne doit pas être vide")

    featurizer(sequences[: min(8, len(sequences))])
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        featurizer(sequences)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations) * 1000.0 / len(sequences))


def benchmark_torch_forward(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    device: str | torch.device = "cpu",
    warmup: int = 100,
    repeats: int = 2000,
) -> float:
    """Retourne la latence médiane du forward pour un échantillon, en ms."""
    device = torch.device(device)
    model = model.to(device).eval()
    example_input = example_input[:1].to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(example_input)
        _synchronize(device)

        durations = np.empty(repeats, dtype=np.float64)
        for run in range(repeats):
            start = time.perf_counter()
            model(example_input)
            _synchronize(device)
            durations[run] = time.perf_counter() - start

    return float(np.median(durations) * 1000.0)


def model_storage_mb(model: torch.nn.Module) -> float:
    """Taille sérialisée du state_dict, en mégaoctets décimaux."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1_000_000.0


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
