"""Deterministic DNA features used by the compact EEIA bioAI student.

The feature order intentionally matches ``day3/src/advanced_features.py``:

* global k-mer frequencies for k = 1..4 (340 values),
* codon frequencies for the three reading phases on both strands (384 values),
* start/stop/open-reading-frame summaries on both strands (24 values),
* base frequencies by phase on the direct strand (12 values),
* one period-three GC summary.

The resulting vector has exactly 761 dimensions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
START_CODONS = frozenset({"ATG", "GTG", "TTG"})
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})
ALLOWED_BASES = frozenset("ACGTN")

FEATURE_DIM = 761
FEATURE_VERSION = "codon-phase-761-v1"

FloatArray = NDArray[np.float32]


def normalize_sequence(sequence: str) -> str:
    """Normalize a DNA sequence and reject unsupported symbols.

    Whitespace is removed and lower-case characters are upper-cased. ``N`` is
    accepted as an unknown base; k-mers containing it are ignored, consistently
    with the original workshop implementation.
    """

    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("sequence must not be empty")
    invalid = sorted(set(normalized) - ALLOWED_BASES)
    if invalid:
        raise ValueError(f"unsupported DNA symbols: {', '.join(invalid)}")
    return normalized


def reverse_complement(sequence: str) -> str:
    """Return the reverse-complement of a DNA sequence."""

    normalized = normalize_sequence(sequence)
    return normalized.translate(COMPLEMENT)[::-1]


def canonical_sequence(sequence: str) -> str:
    """Return an orientation-independent representation of a sequence."""

    normalized = normalize_sequence(sequence)
    reverse = normalized.translate(COMPLEMENT)[::-1]
    return min(normalized, reverse)


@lru_cache(maxsize=1024)
def _kmer_index(kmer: str) -> int | None:
    value = 0
    for nucleotide in kmer:
        index = BASE_TO_INDEX.get(nucleotide)
        if index is None:
            return None
        value = value * 4 + index
    return value


def global_kmer_features(
    sequence: str,
    ks: Sequence[int] = (1, 2, 3, 4),
) -> FloatArray:
    """Compute normalized overlapping k-mer frequencies."""

    normalized = normalize_sequence(sequence)
    blocks: list[FloatArray] = []
    for k in ks:
        if k <= 0:
            raise ValueError("all k values must be positive")
        counts = np.zeros(4**k, dtype=np.float32)
        valid_count = 0
        for position in range(len(normalized) - k + 1):
            index = _kmer_index(normalized[position : position + k])
            if index is not None:
                counts[index] += 1.0
                valid_count += 1
        if valid_count:
            counts /= np.float32(valid_count)
        blocks.append(counts)
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _strand_phase_features(sequence: str) -> tuple[FloatArray, FloatArray]:
    codon_frequencies = np.zeros((3, 64), dtype=np.float32)
    summaries: list[float] = []

    for phase in range(3):
        codons = [
            sequence[position : position + 3] for position in range(phase, len(sequence) - 2, 3)
        ]
        valid_codons = 0
        for codon in codons:
            index = _kmer_index(codon)
            if index is not None:
                codon_frequencies[phase, index] += 1.0
                valid_codons += 1
        if valid_codons:
            codon_frequencies[phase] /= np.float32(valid_codons)

        n_codons = max(len(codons), 1)
        start_fraction = sum(codon in START_CODONS for codon in codons) / n_codons
        stop_fraction = sum(codon in STOP_CODONS for codon in codons) / n_codons

        longest_without_stop = 0
        current_without_stop = 0
        first_stop = len(codons)
        for index, codon in enumerate(codons):
            if codon in STOP_CODONS:
                first_stop = min(first_stop, index)
                longest_without_stop = max(longest_without_stop, current_without_stop)
                current_without_stop = 0
            else:
                current_without_stop += 1
        longest_without_stop = max(longest_without_stop, current_without_stop)

        summaries.extend(
            (
                start_fraction,
                stop_fraction,
                longest_without_stop / n_codons,
                first_stop / n_codons,
            )
        )

    return (
        codon_frequencies.ravel(),
        np.asarray(summaries, dtype=np.float32),
    )


def biological_phase_features(sequence: str) -> FloatArray:
    """Compute reading-frame features on both DNA orientations."""

    normalized = normalize_sequence(sequence)
    direct_codons, direct_summary = _strand_phase_features(normalized)
    reverse_codons, reverse_summary = _strand_phase_features(normalized.translate(COMPLEMENT)[::-1])

    phase_base_frequencies = np.zeros((3, 4), dtype=np.float32)
    for phase in range(3):
        positions = normalized[phase::3]
        valid_count = 0
        for nucleotide in positions:
            index = BASE_TO_INDEX.get(nucleotide)
            if index is not None:
                phase_base_frequencies[phase, index] += 1.0
                valid_count += 1
        if valid_count:
            phase_base_frequencies[phase] /= np.float32(valid_count)

    gc_by_phase = (
        phase_base_frequencies[:, BASE_TO_INDEX["G"]]
        + phase_base_frequencies[:, BASE_TO_INDEX["C"]]
    )
    period_three_gc = np.asarray(
        [gc_by_phase.max() - gc_by_phase.min()],
        dtype=np.float32,
    )

    return np.concatenate(
        (
            direct_codons,
            reverse_codons,
            direct_summary,
            reverse_summary,
            phase_base_frequencies.ravel(),
            period_three_gc,
        )
    ).astype(np.float32, copy=False)


def advanced_features(sequence: str) -> FloatArray:
    """Return the 761-D feature vector for one DNA sequence."""

    normalized = normalize_sequence(sequence)
    features = np.concatenate(
        (
            global_kmer_features(normalized),
            biological_phase_features(normalized),
        )
    ).astype(np.float32, copy=False)
    if features.shape != (FEATURE_DIM,):
        raise RuntimeError(
            f"feature contract violated: expected {(FEATURE_DIM,)}, got {features.shape}"
        )
    return features


def advanced_feature_matrix(sequences: Iterable[str]) -> FloatArray:
    """Build an ``(n, 761)`` feature matrix.

    The function preallocates the output matrix to avoid the repeated temporary
    arrays created by ``np.stack`` on large inputs.
    """

    materialized = list(sequences)
    if not materialized:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    matrix = np.empty((len(materialized), FEATURE_DIM), dtype=np.float32)
    for row, sequence in enumerate(materialized):
        matrix[row] = advanced_features(sequence)
    return matrix
