# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
build_dataset.py — turn raw bacterial genomes (FASTA + GFF) into a labeled
coding (CDS) vs non-coding (intergenic) window dataset.

Usage
-----
python build_dataset.py \
    --raw_dir ../data/raw \
    --out_dir ../data/processed \
    --window 200 --stride 200 --max_per_class_per_genome 4000

Expects, per split (train/val/test):
    raw_dir/<split>/<genome>.fasta
    raw_dir/<split>/<genome>.gff        (same basename, either a minimal
                                          "verified"-source GFF or standard
                                          NCBI GFF3 — both are plain tab-
                                          separated GFF with CDS rows in
                                          column 3, so one parser covers both)

Writes, per split:
    out_dir/<split>.csv   with columns:
        id, split, organism, seqid, start, end, strand, label, sequence
    (label: 1 = coding/CDS window, 0 = non-coding/intergenic window)
"""

import argparse
import csv
import os
import random
from pathlib import Path

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def read_fasta(path: str) -> dict:
    """Minimal multi-record FASTA reader -> {seqid: sequence (upper, no gaps)}."""
    seqs, seqid, chunks = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if seqid is not None:
                    seqs[seqid] = "".join(chunks).upper()
                seqid = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if seqid is not None:
            seqs[seqid] = "".join(chunks).upper()
    return seqs


def read_cds_intervals(path: str) -> dict:
    """Parse GFF (minimal or standard GFF3) -> {seqid: [(start0, end, strand), ...]}.

    Only uses columns 1 (seqid), 3 (type), 4 (start, 1-based inclusive),
    5 (end, inclusive), 7 (strand) — present in both GFF flavors in this
    project. Converts to 0-based half-open [start0, end) for slicing.
    """
    intervals = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            seqid, _source, ftype, start, end, _score, strand = cols[0:7]
            if ftype != "CDS":
                continue
            start0 = int(start) - 1
            end_ = int(end)
            intervals.setdefault(seqid, []).append((start0, end_, strand))
    return intervals


def merge_intervals(intervals):
    """Merge overlapping (start, end) pairs, ignoring strand (union coverage)."""
    if not intervals:
        return []
    spans = sorted((s, e) for s, e, _ in intervals)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def gaps_between(merged_spans, seq_len):
    """Complement of merged CDS spans within [0, seq_len) -> intergenic gaps."""
    gaps, cursor = [], 0
    for s, e in merged_spans:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < seq_len:
        gaps.append((cursor, seq_len))
    return gaps


def clean_enough(window: str, max_n_frac: float = 0.05) -> bool:
    if len(window) == 0:
        return False
    n_count = window.count("N")
    return (n_count / len(window)) <= max_n_frac


def windows_from_span(seq, start, end, window, stride):
    pos = start
    while pos + window <= end:
        yield pos, pos + window
        pos += stride


def build_split(raw_split_dir: Path, window: int, stride: int,
                 max_per_class_per_genome: int, seed: int):
    rng = random.Random(seed)
    rows = []

    fasta_files = sorted(raw_split_dir.glob("*.fasta"))
    for fasta_path in fasta_files:
        gff_path = fasta_path.with_suffix(".gff")
        if not gff_path.exists():
            print(f"  [skip] no matching GFF for {fasta_path.name}")
            continue

        organism = fasta_path.stem
        seqs = read_fasta(str(fasta_path))
        cds_by_seqid = read_cds_intervals(str(gff_path))

        coding_rows, noncoding_rows = [], []

        for seqid, seq in seqs.items():
            cds_intervals = cds_by_seqid.get(seqid, [])

            # coding windows: slide inside each CDS interval, respecting strand
            for start, end, strand in cds_intervals:
                for w_start, w_end in windows_from_span(seq, start, end, window, stride):
                    raw = seq[w_start:w_end]
                    if not clean_enough(raw):
                        continue
                    win_seq = reverse_complement(raw) if strand == "-" else raw
                    coding_rows.append((seqid, w_start, w_end, strand, 1, win_seq))

            # non-coding windows: slide inside intergenic gaps
            merged = merge_intervals(cds_intervals)
            for g_start, g_end in gaps_between(merged, len(seq)):
                for w_start, w_end in windows_from_span(seq, g_start, g_end, window, stride):
                    raw = seq[w_start:w_end]
                    if not clean_enough(raw):
                        continue
                    noncoding_rows.append((seqid, w_start, w_end, "+", 0, raw))

        rng.shuffle(coding_rows)
        rng.shuffle(noncoding_rows)
        n = min(len(coding_rows), len(noncoding_rows), max_per_class_per_genome)
        coding_rows, noncoding_rows = coding_rows[:n], noncoding_rows[:n]

        print(f"  {organism}: {len(coding_rows)} coding + {len(noncoding_rows)} "
              f"non-coding windows (balanced to {n} each)")

        for seqid, w_start, w_end, strand, label, win_seq in coding_rows + noncoding_rows:
            rows.append((organism, seqid, w_start, w_end, strand, label, win_seq))

    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", required=True,
                     help="Path to data/supervised/raw (contains train/val/test)")
    ap.add_argument("--out_dir", required=True,
                     help="Path to write <split>.csv files")
    ap.add_argument("--window", type=int, default=200, help="Window length in bp")
    ap.add_argument("--stride", type=int, default=200,
                     help="Stride between windows (== window means no overlap)")
    ap.add_argument("--max_per_class_per_genome", type=int, default=4000,
                     help="Cap windows per class per genome (class-balanced)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            print(f"[{split}] no such directory: {split_dir}, skipping")
            continue

        print(f"[{split}] building windows from {split_dir} ...")
        rows = build_split(split_dir, args.window, args.stride,
                            args.max_per_class_per_genome, args.seed)

        out_path = out_dir / f"{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "split", "organism", "seqid", "start", "end",
                              "strand", "label", "sequence"])
            for i, (organism, seqid, start, end, strand, label, seq) in enumerate(rows):
                writer.writerow([f"{split}_{i}", split, organism, seqid, start, end,
                                  strand, label, seq])

        n_pos = sum(1 for r in rows if r[5] == 1)
        n_neg = len(rows) - n_pos
        print(f"[{split}] wrote {len(rows)} windows ({n_pos} coding / {n_neg} "
              f"non-coding) -> {out_path}")


if __name__ == "__main__":
    main()
