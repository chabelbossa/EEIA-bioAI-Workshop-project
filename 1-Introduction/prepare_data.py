# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
prepare_data.py — organizer-only, run ONCE before the school starts.

Wires together the two data-prep steps from `../../code/src/` into a single, idempotent
command, so the whole week's data is built/extracted once and never touched again:

  1. build_dataset.py           raw FASTA/GFF  -> labeled window CSVs
  2. extract_evo2_embeddings.py labeled CSVs   -> Evo2 embeddings (needs NVIDIA_API_KEY)

Each step is skipped automatically if its output already exists, so re-running this
script after a partial/failed run is safe and won't recompute what's already there.
Pass --force to rebuild everything regardless.

Usage
-----
    python prepare_data.py                       # build + extract with defaults
    python prepare_data.py --max_per_split 3000  # cap embeddings API cost/time
    python prepare_data.py --skip_embeddings      # only (re)build the CSVs
    python prepare_data.py --force                # ignore existing outputs, redo all

See ../../facilitator_guide/day_by_day.md ("Before Day 0") for the full checklist this
script automates the data-heavy part of.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC_DIR = REPO_ROOT / "code" / "src"
RAW_DIR = REPO_ROOT / "data" / "supervised" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "supervised" / "processed"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "supervised" / "embeddings"


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def build_dataset(window: int, stride: int, force: bool):
    splits_done = all((PROCESSED_DIR / f"{s}.csv").exists() for s in ("train", "val", "test"))
    if splits_done and not force:
        print(f"[build_dataset] {PROCESSED_DIR} already has train/val/test.csv, skipping "
              f"(use --force to rebuild)")
        return
    run([
        sys.executable, str(SRC_DIR / "build_dataset.py"),
        "--raw_dir", str(RAW_DIR),
        "--out_dir", str(PROCESSED_DIR),
        "--window", str(window),
        "--stride", str(stride),
    ])


def extract_embeddings(max_per_split, target_layer: str, model: str, force: bool):
    splits_done = all((EMBEDDINGS_DIR / f"{s}.npz").exists() for s in ("train", "val", "test"))
    if splits_done and not force:
        print(f"[extract_evo2_embeddings] {EMBEDDINGS_DIR} already has train/val/test.npz, "
              f"skipping (use --force to rebuild)")
        return
    cmd = [
        sys.executable, str(SRC_DIR / "extract_evo2_embeddings.py"),
        "--processed_dir", str(PROCESSED_DIR),
        "--out_dir", str(EMBEDDINGS_DIR),
        "--target_layer", target_layer,
        "--model", model,
    ]
    if max_per_split:
        cmd += ["--max_per_split", str(max_per_split)]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--max_per_split", type=int, default=3000,
                     help="Class-balanced subsample per split before Evo2 extraction "
                          "(API cost/time knob)")
    ap.add_argument("--target_layer", default="blocks.26")
    ap.add_argument("--model", default="evo2-7b")
    ap.add_argument("--skip_embeddings", action="store_true",
                     help="Only (re)build the labeled CSVs, don't call the Evo2 API")
    ap.add_argument("--force", action="store_true",
                     help="Rebuild/re-extract even if outputs already exist")
    args = ap.parse_args()

    build_dataset(args.window, args.stride, args.force)

    if args.skip_embeddings:
        print("[extract_evo2_embeddings] --skip_embeddings passed, not calling the Evo2 API")
        return

    import os
    if not os.environ.get("NVIDIA_API_KEY"):
        raise SystemExit(
            "NVIDIA_API_KEY is not set — export it first, or pass --skip_embeddings to "
            "only build the CSVs for now and run the embedding step later."
        )
    extract_embeddings(args.max_per_split, args.target_layer, args.model, args.force)

    print("\nData prep done. Students will consume:")
    print(f"  {PROCESSED_DIR}/{{train,val,test}}.csv")
    print(f"  {EMBEDDINGS_DIR}/{{train,val,test}}.npz")


if __name__ == "__main__":
    main()
