"""Caractéristiques compactes pour distinguer ADN codant et non codant.

Le vecteur combine des fréquences globales de k-mers avec des statistiques liées
aux trois cadres de lecture, calculées dans les deux orientations de l'ADN.
"""

import numpy as np


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
START_CODONS = {"ATG", "GTG", "TTG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}


def reverse_complement(sequence: str) -> str:
    """Retourne le reverse-complement d'une séquence d'ADN."""
    return sequence.translate(COMPLEMENT)[::-1]


def canonical_sequence(sequence: str) -> str:
    """Forme indépendante de l'orientation, utile pour auditer les doublons."""
    reverse = reverse_complement(sequence)
    return min(sequence, reverse)


def _kmer_index(kmer: str):
    value = 0
    for nucleotide in kmer:
        index = BASE_TO_INDEX.get(nucleotide)
        if index is None:
            return None
        value = value * 4 + index
    return value


def global_kmer_features(sequence: str, ks=(1, 2, 3, 4)) -> np.ndarray:
    """Fréquences normalisées des k-mers chevauchants pour plusieurs valeurs de k."""
    blocks = []
    for k in ks:
        counts = np.zeros(4**k, dtype=np.float32)
        valid_count = 0
        for position in range(len(sequence) - k + 1):
            index = _kmer_index(sequence[position : position + k])
            if index is not None:
                counts[index] += 1.0
                valid_count += 1
        if valid_count:
            counts /= valid_count
        blocks.append(counts)
    return np.concatenate(blocks)


def _strand_phase_features(sequence: str):
    """Composition en codons et résumés des trois phases d'une orientation."""
    codon_frequencies = np.zeros((3, 64), dtype=np.float32)
    summaries = []

    for phase in range(3):
        codons = [
            sequence[position : position + 3]
            for position in range(phase, len(sequence) - 2, 3)
        ]
        valid_codons = 0
        for codon in codons:
            index = _kmer_index(codon)
            if index is not None:
                codon_frequencies[phase, index] += 1.0
                valid_codons += 1
        if valid_codons:
            codon_frequencies[phase] /= valid_codons

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
            [
                start_fraction,
                stop_fraction,
                longest_without_stop / n_codons,
                first_stop / n_codons,
            ]
        )

    return codon_frequencies.ravel(), np.asarray(summaries, dtype=np.float32)


def biological_phase_features(sequence: str) -> np.ndarray:
    """Caractéristiques des cadres de lecture direct et reverse-complement."""
    direct_codons, direct_summary = _strand_phase_features(sequence)
    reverse_codons, reverse_summary = _strand_phase_features(
        reverse_complement(sequence)
    )

    phase_base_frequencies = np.zeros((3, 4), dtype=np.float32)
    for phase in range(3):
        positions = sequence[phase::3]
        valid_count = 0
        for nucleotide in positions:
            index = BASE_TO_INDEX.get(nucleotide)
            if index is not None:
                phase_base_frequencies[phase, index] += 1.0
                valid_count += 1
        if valid_count:
            phase_base_frequencies[phase] /= valid_count

    gc_by_phase = (
        phase_base_frequencies[:, BASE_TO_INDEX["G"]]
        + phase_base_frequencies[:, BASE_TO_INDEX["C"]]
    )
    period_three_gc = np.asarray(
        [gc_by_phase.max() - gc_by_phase.min()], dtype=np.float32
    )

    return np.concatenate(
        [
            direct_codons,
            reverse_codons,
            direct_summary,
            reverse_summary,
            phase_base_frequencies.ravel(),
            period_three_gc,
        ]
    )


def advanced_feature_matrix(sequences) -> np.ndarray:
    """Matrice 761-D : k-mers 1–4 et caractéristiques biologiques de phase."""
    return np.stack(
        [
            np.concatenate(
                [global_kmer_features(sequence), biological_phase_features(sequence)]
            )
            for sequence in sequences
        ]
    ).astype(np.float32)
