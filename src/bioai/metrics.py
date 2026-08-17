"""Binary classification metrics with explicit thresholding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    brier: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _optional_ranking_metric(function, labels: np.ndarray, probabilities: np.ndarray):
    if np.unique(labels).size < 2:
        return None
    return float(function(labels, probabilities))


def evaluate_binary_probabilities(
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError("labels and probabilities must have the same length")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("labels must be binary values 0/1")
    if not np.isfinite(y_score).all() or ((y_score < 0) | (y_score > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    predictions = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=_optional_ranking_metric(roc_auc_score, y_true, y_score),
        average_precision=_optional_ranking_metric(
            average_precision_score,
            y_true,
            y_score,
        ),
        brier=float(brier_score_loss(y_true, y_score)),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )
