import numpy as np
import pytest

from bioai.features import (
    FEATURE_DIM,
    advanced_feature_matrix,
    advanced_features,
    canonical_sequence,
    global_kmer_features,
    normalize_sequence,
    reverse_complement,
)


def test_normalize_and_reverse_complement() -> None:
    assert normalize_sequence(" atg cN\n") == "ATGCN"
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("ANNT") == "ANNT"


def test_canonical_sequence_is_orientation_independent() -> None:
    sequence = "ATGCCGTTA"
    assert canonical_sequence(sequence) == canonical_sequence(reverse_complement(sequence))


def test_global_kmer_features_are_normalized_per_block() -> None:
    vector = global_kmer_features("AAAA", ks=(1, 2))
    one_mer = vector[:4]
    two_mer = vector[4:]
    assert one_mer[0] == pytest.approx(1.0)
    assert two_mer[0] == pytest.approx(1.0)
    assert one_mer.sum() == pytest.approx(1.0)
    assert two_mer.sum() == pytest.approx(1.0)


def test_advanced_features_contract() -> None:
    sequence = "ATG" * 66 + "AT"
    vector = advanced_features(sequence)
    assert vector.shape == (FEATURE_DIM,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_matrix_contract_and_empty_input() -> None:
    sequences = ["ATG" * 66 + "AT", "TAA" * 66 + "TA"]
    matrix = advanced_feature_matrix(sequences)
    assert matrix.shape == (2, FEATURE_DIM)
    assert advanced_feature_matrix([]).shape == (0, FEATURE_DIM)


def test_invalid_symbols_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported DNA symbols"):
        advanced_features("ATGX")
