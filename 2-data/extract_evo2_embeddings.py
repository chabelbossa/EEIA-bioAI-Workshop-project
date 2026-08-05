# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
NVIDIA-API-KEY: https://build.nvidia.com/arc/evo2-7b-forward/modelcard

extract_evo2_embeddings.py — ORGANIZER-run.

Calls the NVIDIA NIM Evo2 "forward" API on the labeled windows produced by
build_dataset.py, mean-pools the per-token activations of one layer into a
single embedding per window, and saves embeddings + labels for the
supervised teacher classifier (notebook 02) and later distillation
(notebook 03).

Students do NOT run this during the week — the API key/quota is yours, and
this is exactly the "pre-extract before the school" step that makes the
core pipeline independent of live API availability.

Request/response format:
    POST {FORWARD_URL} json={"sequence": seq, "output_layers": [layer, ...]}
    -> {"data": base64(npz bytes)} -> npz["<layer>.output"], shape (1, seq_len, hidden_dim)
    (batch dim, if present, is index 0 and gets dropped)

Several layers can be requested in ONE call — they come from the same forward
pass, so extra layers cost no extra API request. Useful for comparing where in
the network the coding/non-coding signal is most linearly decodable.
On evo2-7b the blocks are 0-indexed 0..31 (blocks.31 is the last).

Usage
-----
export NVIDIA_API_KEY=nvapi-...
python extract_evo2_embeddings.py \
    --processed_dir ./processed \
    --out_dir ./embeddings \
    --max_per_split train=4000 val=1000 test=1000 \
    --target_layers blocks.26 blocks.31

--max_per_split also accepts a single number for every split (e.g. 3000).
Budget it: ~4s per window, so 6000 windows is roughly 7 hours.
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FORWARD_URL_TMPL = "https://health.api.nvidia.com/v1/biology/arc/{model}/forward"
RATE_LIMIT_WAIT = 120
MAX_RETRIES = 10


def get_activations(sequence: str, layers, api_key: str, model: str) -> dict:
    """Call the NIM forward endpoint once and return {layer: (seq_len, hidden_dim)}
    for every requested layer, retrying on rate limits / transient errors.

    Asking for several layers in a single request costs no extra call — the API
    returns them all in the same npz archive.
    """
    url = FORWARD_URL_TMPL.format(model=model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                url, headers=headers,
                json={"sequence": sequence, "output_layers": list(layers)},
                timeout=600,
            )
        except requests.exceptions.RequestException as exc:
            # dropped connection / read timeout / DNS hiccup: no status code to
            # inspect, but just as retryable as a 503 — this is what kills long
            # unattended runs if left uncaught
            wait = 30 * (attempt + 1)
            print(f"  [réseau] {type(exc).__name__} — nouvel essai dans {wait}s "
                  f"(tentative {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        if r.status_code in (429, 503, 400, 504):
            wait = RATE_LIMIT_WAIT * (attempt + 1) if r.status_code != 504 else 30 * (attempt + 1)
            print(f"  [{r.status_code}] retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        r.raise_for_status()
        payload = r.json()
        decoded = base64.b64decode(payload["data"].encode("ascii"))
        npz = np.load(io.BytesIO(decoded))

        out = {}
        for layer in layers:
            # the API names the archive entry "<layer>.output" (not "<layer>")
            key = f"{layer}.output" if f"{layer}.output" in npz.files else layer
            if key not in npz.files:
                raise KeyError(
                    f"layer {layer!r} not in response; available keys: {list(npz.files)}"
                )
            emb = npz[key]
            if emb.ndim == 3:  # drop the batch dim if present
                emb = emb[0]
            out[layer] = emb
        return out

    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries calling Evo2 NIM API")


def parse_max_per_split(values, splits) -> dict:
    """Turn --max_per_split into {split: limit or None}.

    Accepts a single number applied to every split ("3000") or per-split
    assignments ("train=4000 val=1000 test=1000"); splits left unassigned are
    extracted in full.
    """
    if not values:
        return {s: None for s in splits}

    if len(values) == 1 and "=" not in values[0]:
        return {s: int(values[0]) for s in splits}

    limits = {s: None for s in splits}
    for item in values:
        if "=" not in item:
            raise SystemExit(
                f"--max_per_split: expected split=number, got {item!r} "
                "(mixing a bare number with split=number is not allowed)"
            )
        split, _, raw = item.partition("=")
        if split not in limits:
            raise SystemExit(
                f"--max_per_split: unknown split {split!r}, expected one of {splits}"
            )
        limits[split] = int(raw)
    return limits


def load_checkpoint(ckpt_path, run_id):
    """Return the resume index, but only if the checkpoint belongs to an
    identical run (same subsample size, same layers, same model).

    Resuming across a changed --max_per_split would silently mix embeddings
    from two different subsamples: the index points into the *old* sample
    while the DataFrame is the new one. Better to restart the split.
    """
    if not os.path.exists(ckpt_path):
        return 0

    with open(ckpt_path) as f:
        ckpt = json.load(f)

    if "run_id" not in ckpt:
        # checkpoint written before run_id existed: no way to verify it, but
        # discarding it would throw away hours of extraction — trust it
        print("  checkpoint sans run_id (ancienne version) — repris tel quel")
        return ckpt["next_index"]

    if ckpt["run_id"] != run_id:
        print(f"  checkpoint ignoré : il vient d'un run différent "
              f"({ckpt.get('run_id')} != {run_id}) — ce split repart de zéro")
        return 0
    return ckpt["next_index"]


def save_checkpoint(ckpt_path, next_index, run_id):
    with open(ckpt_path, "w") as f:
        json.dump({"next_index": next_index, "run_id": run_id}, f)


def make_run_id(n: int, layers, model: str) -> str:
    """Identity of a run: anything that changes what a row index means."""
    return f"n={n}|layers={','.join(layers)}|model={model}"


def extract_split(df: pd.DataFrame, out_dir: Path, split: str,
                   layers, api_key: str, model: str, force: bool = False):
    """Extract one mean-pooled embedding per window, for every requested layer.

    All layers come from the same forward pass, so asking for several costs
    nothing extra. They are written to the same <split>.npz under the keys
    "embeddings_<layer>", alongside the shared labels/ids. The first layer is
    also written as plain "embeddings" so the default path stays simple.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split}.npz"
    ckpt_path = out_dir / f"{split}_checkpoint.json"

    def partial_path(layer):
        return out_dir / f"{split}_partial_{layer}.npy"

    n = len(df)
    if n == 0:
        print(f"[{split}] 0 windows to extract, skipping "
              "(check --max_per_split / the split's CSV)")
        return

    if npz_path.exists() and not force:
        print(f"[{split}] {npz_path} existe déjà — split ignoré "
              "(--force pour le réextraire)")
        return

    run_id = make_run_id(n, layers, model)
    start = load_checkpoint(ckpt_path, run_id)
    if start and not all(partial_path(layer).exists() for layer in layers):
        print(f"  [{split}] fichiers partiels manquants — ce split repart de zéro")
        start = 0

    if start == 0:
        embeddings = None  # one matrix per layer, allocated once we know hidden_dim
    else:
        embeddings = {layer: np.load(partial_path(layer)) for layer in layers}
        print(f"  [{split}] reprise à la fenêtre {start}/{n}")

    for i in range(start, n):
        seq = df.iloc[i]["sequence"]
        acts = get_activations(seq, layers, api_key, model)  # {layer: (seq_len, hidden)}
        pooled = {layer: a.mean(axis=0) for layer, a in acts.items()}  # -> (hidden,)

        if embeddings is None:
            embeddings = {
                layer: np.zeros((n, p.shape[0]), dtype=np.float32)
                for layer, p in pooled.items()
            }
        for layer, vec in pooled.items():
            embeddings[layer][i] = vec

        if (i + 1) % 50 == 0 or i == n - 1:
            for layer, mat in embeddings.items():
                np.save(partial_path(layer), mat)
            save_checkpoint(ckpt_path, i + 1, run_id)
            print(f"  [{split}] {i + 1}/{n} embeddings extracted", end="\r")

    print()
    arrays = {f"embeddings_{layer}": mat for layer, mat in embeddings.items()}
    arrays["embeddings"] = embeddings[layers[0]]  # default layer
    np.savez(
        npz_path,
        labels=df["label"].to_numpy(),
        ids=df["id"].to_numpy(),
        layers=np.array(list(layers)),
        **arrays,
    )
    for layer in layers:
        os.remove(partial_path(layer))
    os.remove(ckpt_path)
    shapes = ", ".join(f"{layer}: {mat.shape}" for layer, mat in embeddings.items())
    print(f"[{split}] saved {shapes} -> {npz_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed_dir", required=True,
                     help="Path to data/supervised/processed (build_dataset.py output)")
    ap.add_argument("--out_dir", required=True,
                     help="Path to write <split>.npz (embeddings + labels + ids)")
    ap.add_argument("--target_layers", nargs="+", default=["blocks.26"],
                     help="One or more layers to extract, e.g. --target_layers "
                          "blocks.26 blocks.31. They all come from the SAME forward "
                          "pass, so extra layers cost no extra API call. The first "
                          "one is the default used by the notebooks.")
    ap.add_argument("--model", default="evo2-7b")
    ap.add_argument("--max_per_split", nargs="+", default=None,
                     help="Class-balanced subsample before extraction. Either one "
                          "value for every split (--max_per_split 3000) or a value "
                          "per split (--max_per_split train=4000 val=1000 test=1000). "
                          "This is the main API cost/time knob: ~4s per window.")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--force", action="store_true",
                     help="Re-extract splits whose <split>.npz already exists "
                          "(by default they are skipped, so a relaunch only "
                          "finishes what is missing)")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("Set NVIDIA_API_KEY in your environment first.")

    max_per_split = parse_max_per_split(args.max_per_split, args.splits)

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    for split in args.splits:
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[{split}] {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path, dtype={"sequence": str})
        limit = max_per_split[split]
        # `limit is not None`, not `if limit` — 0 means "extract nothing", not "no limit"
        if limit is not None and len(df) > limit:
            n_classes = df["label"].nunique()
            per_class = limit // n_classes
            if per_class == 0:
                print(f"[{split}] --max_per_split {limit} is below the number of "
                      f"classes ({n_classes}) — nothing to extract for this split")
            df = (
                df.groupby("label", group_keys=False)[df.columns.tolist()]
                .apply(lambda g: g.sample(min(len(g), per_class), random_state=42))
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

        print(f"[{split}] extracting Evo2 embeddings for {len(df)} windows "
              f"(layers: {', '.join(args.target_layers)}) ...")
        extract_split(df, out_dir, split, args.target_layers, api_key, args.model,
                       force=args.force)


if __name__ == "__main__":
    main()
