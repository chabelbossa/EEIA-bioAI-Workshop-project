"""Compact student model and portable checkpoint helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .features import FEATURE_DIM, FEATURE_VERSION


@dataclass(frozen=True, slots=True)
class StudentConfig:
    input_dim: int = FEATURE_DIM
    hidden_dim: int = 64
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class StudentMLP(nn.Module):
    """The 48,961-parameter MLP used by the final OOF experiment."""

    def __init__(self, config: StudentConfig | None = None) -> None:
        super().__init__()
        self.config = config or StudentConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_dim:
            raise ValueError(
                f"expected inputs shaped (batch, {self.config.input_dim}), "
                f"got {tuple(inputs.shape)}"
            )
        return self.net(inputs).squeeze(-1)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_probabilities(
    model: StudentMLP,
    features: NDArray[np.float32] | torch.Tensor,
    *,
    device: str | torch.device = "auto",
) -> NDArray[np.float32]:
    """Run deterministic batched inference and return class-1 probabilities."""

    resolved = resolve_device(device)
    tensor = (
        features
        if isinstance(features, torch.Tensor)
        else torch.as_tensor(features, dtype=torch.float32)
    )
    model = model.to(resolved).eval()
    with torch.inference_mode():
        logits = model(tensor.to(resolved))
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return probabilities


def save_checkpoint(
    path: str | Path,
    model: StudentMLP,
    *,
    threshold: float = 0.5,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    payload = {
        "format_version": 1,
        "feature_version": FEATURE_VERSION,
        "model_config": asdict(model.config),
        "threshold": float(threshold),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[StudentMLP, float, dict[str, Any]]:
    """Load a versioned student checkpoint.

    Raw state dictionaries are intentionally rejected: a public artifact must
    carry its architecture, feature contract and threshold.
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    required = {"format_version", "feature_version", "model_config", "state_dict"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("unsupported checkpoint: expected the versioned bioAI format")
    if payload["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            "checkpoint feature version mismatch: "
            f"{payload['feature_version']!r} != {FEATURE_VERSION!r}"
        )
    config = StudentConfig(**payload["model_config"])
    model = StudentMLP(config)
    model.load_state_dict(payload["state_dict"])
    threshold = float(payload.get("threshold", 0.5))
    metadata = dict(payload.get("metadata", {}))
    return model, threshold, metadata
