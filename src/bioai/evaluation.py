"""End-to-end evaluation with an explicit canonically-unseen surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import audit_sequence_overlap, canonically_unseen_mask
from .features import advanced_feature_matrix
from .metrics import BinaryMetrics, evaluate_binary_probabilities
from .model import load_checkpoint, predict_probabilities


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    checkpoint: str
    total_rows: int
    canonically_unseen_rows: int
    overlap_audit: dict[str, int]
    all_rows: BinaryMetrics
    canonically_unseen: BinaryMetrics | None
    checkpoint_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["all_rows"] = self.all_rows.to_dict()
        payload["canonically_unseen"] = (
            None if self.canonically_unseen is None else self.canonically_unseen.to_dict()
        )
        return payload


def evaluate_checkpoint_on_csv(
    *,
    checkpoint: str | Path,
    train_csv: str | Path,
    evaluation_csv: str | Path,
    sequence_column: str = "sequence",
    label_column: str = "label",
    threshold: float | None = None,
    device: str = "auto",
) -> EvaluationReport:
    """Evaluate a versioned checkpoint on all rows and the unseen subset.

    The function does not choose a threshold: it uses the checkpoint threshold
    unless one is supplied explicitly. Calling it on the test split is therefore
    appropriate only after the pipeline has been frozen.
    """

    train_frame = pd.read_csv(train_csv, dtype={sequence_column: str})
    evaluation_frame = pd.read_csv(evaluation_csv, dtype={sequence_column: str})
    for name, frame in (("train", train_frame), ("evaluation", evaluation_frame)):
        missing = {sequence_column, label_column} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
        if frame[sequence_column].isna().any():
            raise ValueError(f"{name} CSV contains missing sequences")

    model, checkpoint_threshold, metadata = load_checkpoint(checkpoint)
    effective_threshold = checkpoint_threshold if threshold is None else threshold

    sequences = evaluation_frame[sequence_column].tolist()
    features = advanced_feature_matrix(sequences)
    probabilities = predict_probabilities(model, features, device=device)
    labels = evaluation_frame[label_column].to_numpy()

    overlap = audit_sequence_overlap(train_frame[sequence_column], sequences)
    unseen_mask = canonically_unseen_mask(train_frame[sequence_column], sequences)
    unseen_metrics = None
    if unseen_mask.any():
        unseen_metrics = evaluate_binary_probabilities(
            labels[unseen_mask],
            probabilities[unseen_mask],
            threshold=effective_threshold,
        )

    return EvaluationReport(
        checkpoint=str(checkpoint),
        total_rows=len(evaluation_frame),
        canonically_unseen_rows=int(unseen_mask.sum()),
        overlap_audit=overlap.to_dict(),
        all_rows=evaluate_binary_probabilities(
            labels,
            probabilities,
            threshold=effective_threshold,
        ),
        canonically_unseen=unseen_metrics,
        checkpoint_metadata=metadata,
    )
