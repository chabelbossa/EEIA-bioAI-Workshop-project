"""Numerically stable utilities for binary knowledge distillation."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def validate_distillation_parameters(alpha: float, temperature: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")


def binary_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Combine hard-label BCE and soft teacher targets.

    The soft term uses ``binary_cross_entropy_with_logits`` for numerical
    stability. Multiplication by ``temperature**2`` follows the conventional
    gradient-rescaling used in knowledge distillation.
    """

    validate_distillation_parameters(alpha, temperature)
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have identical shapes")
    if student_logits.shape != labels.shape:
        raise ValueError("labels must have the same shape as the logits")

    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels.float())
    soft_targets = torch.sigmoid(teacher_logits.detach() / temperature)
    soft_loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        soft_targets,
    ) * (temperature**2)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch without assuming CUDA availability."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
