"""Dataset integrity checks for exact and reverse-complement overlap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .features import canonical_sequence, normalize_sequence

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class OverlapAudit:
    """Summary of train/evaluation sequence overlap."""

    train_rows: int
    evaluation_rows: int
    train_unique_exact: int
    evaluation_unique_exact: int
    exact_overlap_unique: int
    canonical_overlap_unique: int
    evaluation_rows_exact_overlap: int
    evaluation_rows_canonical_overlap: int
    evaluation_rows_canonically_unseen: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _materialize(sequences: Iterable[str]) -> list[str]:
    return [normalize_sequence(sequence) for sequence in sequences]


def canonically_unseen_mask(
    train_sequences: Iterable[str],
    evaluation_sequences: Iterable[str],
) -> BoolArray:
    """Return a mask selecting evaluation rows unseen in either orientation."""

    train_canonical = {canonical_sequence(sequence) for sequence in train_sequences}
    evaluation = _materialize(evaluation_sequences)
    return np.asarray(
        [canonical_sequence(sequence) not in train_canonical for sequence in evaluation],
        dtype=np.bool_,
    )


def audit_sequence_overlap(
    train_sequences: Iterable[str],
    evaluation_sequences: Iterable[str],
) -> OverlapAudit:
    """Audit exact and reverse-complement overlap between two sequence collections."""

    train = _materialize(train_sequences)
    evaluation = _materialize(evaluation_sequences)

    train_exact = set(train)
    evaluation_exact = set(evaluation)
    train_canonical = {canonical_sequence(sequence) for sequence in train}
    evaluation_canonical = {canonical_sequence(sequence) for sequence in evaluation}

    exact_rows = sum(sequence in train_exact for sequence in evaluation)
    canonical_rows = sum(
        canonical_sequence(sequence) in train_canonical for sequence in evaluation
    )

    return OverlapAudit(
        train_rows=len(train),
        evaluation_rows=len(evaluation),
        train_unique_exact=len(train_exact),
        evaluation_unique_exact=len(evaluation_exact),
        exact_overlap_unique=len(train_exact & evaluation_exact),
        canonical_overlap_unique=len(train_canonical & evaluation_canonical),
        evaluation_rows_exact_overlap=exact_rows,
        evaluation_rows_canonical_overlap=canonical_rows,
        evaluation_rows_canonically_unseen=len(evaluation) - canonical_rows,
    )


def audit_csv_files(
    train_csv: str | Path,
    evaluation_csv: str | Path,
    sequence_column: str = "sequence",
) -> OverlapAudit:
    """Load two CSV files and run :func:`audit_sequence_overlap`."""

    train_frame = pd.read_csv(train_csv, dtype={sequence_column: str})
    evaluation_frame = pd.read_csv(evaluation_csv, dtype={sequence_column: str})
    for name, frame in (("train", train_frame), ("evaluation", evaluation_frame)):
        if sequence_column not in frame.columns:
            raise ValueError(f"{name} CSV has no {sequence_column!r} column")
        if frame[sequence_column].isna().any():
            raise ValueError(f"{name} CSV contains missing sequences")
    return audit_sequence_overlap(
        train_frame[sequence_column],
        evaluation_frame[sequence_column],
    )
