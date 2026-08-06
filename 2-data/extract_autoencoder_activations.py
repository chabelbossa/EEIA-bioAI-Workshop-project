# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
NVIDIA-API-KEY : https://build.nvidia.com/arc/evo2-7b-forward/modelcard

extract_autoencoder_activations.py — ORGANISATEUR.

Prépare le jeu de données de la piste bonus (Jour 5, autoencodeur parcimonieux).

Différence avec extract_evo2_embeddings.py :

    extract_evo2_embeddings.py   fenêtres de 200 pb ÉTIQUETÉES, mean-poolées
                                 -> un vecteur par fenêtre, pour le classifieur

    ce script                    longues fenêtres (4096 pb par défaut), SANS
                                 étiquette, activations gardées PAR TOKEN
                                 -> des centaines de milliers de vecteurs bruts,
                                    pour entraîner un SAE non supervisé

On prend de longues fenêtres pour laisser Evo2 exploiter le contexte : c'est là que
naissent les structures qu'on espère voir émerger (périodicité de 3 des codons, etc.).

Sortie (format attendu par load_autoencoder_activations) :

    <out_dir>/<split>/acts.dat    (total_tokens, d_model) float32, brut
    <out_dir>/<split>/meta.json   {"total_tokens", "d_model", "dtype", ...}

Usage
-----
export NVIDIA_API_KEY=nvapi-...
python extract_autoencoder_activations.py \
    --raw_dir ./raw --out_dir ./autoencoder --split train \
    --window 4096 --token_stride 1 --max_windows 20

Coût : une fenêtre de 4096 pb ≈ 40 s d'API, ≈ 46 Mo de réponse et 67 Mo sur disque
(4096 vecteurs de 4096 dimensions, toutes les positions conservées).

On garde TOUTES les positions (token_stride=1) parce que l'analyse du Jour 5 cherche
une périodicité par FFT sur des positions consécutives : sous-échantillonner replierait
le signal de période 3 des codons et donnerait un résultat faux.

Le script est reprenable : relancez-le, il continue là où il s'était arrêté.
"""

import argparse
import base64
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import requests

FORWARD_URL_TMPL = "https://health.api.nvidia.com/v1/biology/arc/{model}/forward"
RATE_LIMIT_WAIT = 120
MAX_RETRIES = 10
VALID_BASES = set("ACGT")


# --------------------------------------------------------------------- lecture
def iter_fasta(path):
    """Rend (nom, séquence) pour chaque enregistrement d'un fichier FASTA."""
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.upper())
    if name is not None:
        yield name, "".join(chunks)


def collect_windows(raw_dir: Path, window: int, max_windows: int, seed: int):
    """Fenêtres longues tirées au hasard dans les génomes, SANS étiquette.

    Aucun GFF n'est lu : pour l'apprentissage non supervisé, la notion de gène
    n'existe pas. On rejette seulement les fenêtres contenant des N.
    """
    fastas = sorted(p for p in raw_dir.rglob("*.fasta"))
    if not fastas:
        raise SystemExit(f"aucun .fasta trouvé sous {raw_dir}")

    rng = random.Random(seed)
    contigs = []
    for f in fastas:
        for name, seq in iter_fasta(f):
            if len(seq) >= window:
                contigs.append((f.name, name, seq))
    if not contigs:
        raise SystemExit(f"aucun contig d'au moins {window} pb")

    print(f"  {len(fastas)} fichiers FASTA, {len(contigs)} contigs exploitables")

    windows, tries = [], 0
    while len(windows) < max_windows and tries < max_windows * 50:
        tries += 1
        fname, cname, seq = contigs[rng.randrange(len(contigs))]
        start = rng.randrange(0, len(seq) - window + 1)
        sub = seq[start:start + window]
        if set(sub) <= VALID_BASES:
            windows.append({"file": fname, "contig": cname, "start": start, "seq": sub})
    if len(windows) < max_windows:
        print(f"  attention : seulement {len(windows)} fenêtres propres trouvées")
    return windows


# --------------------------------------------------------------------- API
def get_activations(sequence: str, layer: str, api_key: str, model: str) -> np.ndarray:
    """Retourne les activations (seq_len, d_model) d'une fenêtre, avec reprises."""
    url = FORWARD_URL_TMPL.format(model=model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers,
                              json={"sequence": sequence, "output_layers": [layer]},
                              timeout=900)
        except requests.exceptions.RequestException as exc:
            wait = 30 * (attempt + 1)
            print(f"  [réseau] {type(exc).__name__} — nouvel essai dans {wait}s "
                  f"({attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        if r.status_code in (429, 503, 400, 504):
            wait = RATE_LIMIT_WAIT * (attempt + 1) if r.status_code != 504 else 30 * (attempt + 1)
            print(f"  [{r.status_code}] nouvel essai dans {wait}s "
                  f"({attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        r.raise_for_status()
        npz = np.load(io.BytesIO(base64.b64decode(r.json()["data"].encode("ascii"))))
        key = f"{layer}.output" if f"{layer}.output" in npz.files else layer
        if key not in npz.files:
            raise KeyError(f"couche {layer!r} absente ; disponibles : {list(npz.files)}")
        acts = npz[key]
        if acts.ndim == 3:
            acts = acts[0]
        return acts

    raise RuntimeError(f"échec après {MAX_RETRIES} tentatives")


# --------------------------------------------------------------------- état
def load_state(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"windows_done": 0, "total_tokens": 0}


def save_meta(meta_path, meta):
    meta_path.write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default="./raw",
                    help="dossier contenant les génomes .fasta")
    ap.add_argument("--out_dir", default="./autoencoder",
                    help="dossier de sortie (un sous-dossier par split)")
    ap.add_argument("--split", default="train", choices=["train", "eval"],
                    help="le Jour 5 attend 'train' et 'eval' (pas 'val')")
    ap.add_argument("--window", type=int, default=4096,
                    help="longueur des fenêtres envoyées à Evo2 (contexte long)")
    ap.add_argument("--token_stride", type=int, default=1,
                    help="on garde 1 token sur N. LAISSEZ 1 : l'analyse de périodicité "
                         "du Jour 5 (FFT sur positions consécutives) est fausse dès "
                         "que l'on sous-échantillonne")
    ap.add_argument("--max_windows", type=int, default=20,
                    help="nombre de fenêtres à extraire pour ce split")
    ap.add_argument("--layer", default="blocks.26")
    ap.add_argument("--model", default="evo2-7b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("Exportez d'abord NVIDIA_API_KEY.")

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_path, meta_path = out_dir / "acts.dat", out_dir / "meta.json"
    state_path = out_dir / "_state.json"

    state = load_state(state_path)
    start = state["windows_done"]

    # la sélection des fenêtres est déterministe (même seed) : la reprise est sûre
    print(f"[{args.split}] sélection des fenêtres ({args.window} pb, sans étiquette)")
    windows = collect_windows(Path(args.raw_dir), args.window, args.max_windows, args.seed)

    if start:
        print(f"  reprise : {start}/{len(windows)} fenêtres déjà extraites "
              f"({state['total_tokens']} vecteurs)")
        if not acts_path.exists():
            print("  acts.dat manquant — on repart de zéro")
            start, state = 0, {"windows_done": 0, "total_tokens": 0}

    if start == 0 and acts_path.exists():
        acts_path.unlink()

    per_window = len(range(0, args.window, args.token_stride))
    print(f"  {len(windows)} fenêtres × {per_window} vecteurs "
          f"= {len(windows) * per_window} vecteurs attendus")

    d_model = state.get("d_model")
    t_start = time.time()

    with open(acts_path, "ab") as fh:
        for i in range(start, len(windows)):
            w = windows[i]
            acts = get_activations(w["seq"], args.layer, api_key, args.model)
            kept = np.ascontiguousarray(acts[::args.token_stride], dtype=np.float32)

            d_model = d_model or kept.shape[1]
            if kept.shape[1] != d_model:
                raise RuntimeError(f"d_model incohérent : {kept.shape[1]} vs {d_model}")

            fh.write(kept.tobytes())
            fh.flush()

            # provenance : de quel génome/contig/position vient chaque bloc de lignes.
            # Sans cela, impossible de savoir qu'une tranche de acts.dat est
            # continue — le fichier est une concaténation de fenêtres disjointes.
            state.setdefault("windows", []).append({
                "row_start": state["total_tokens"],
                "n_rows": int(kept.shape[0]),
                "file": w["file"], "contig": w["contig"],
                "genome_start": int(w["start"]),
                "genome_end": int(w["start"] + args.window),
            })
            state["windows_done"] = i + 1
            state["total_tokens"] += kept.shape[0]
            state["d_model"] = d_model
            state_path.write_text(json.dumps(state))
            save_meta(meta_path, {
                "total_tokens": state["total_tokens"],
                "d_model": d_model,
                "dtype": "float32",
                "split": args.split,
                "layer": args.layer,
                "model": args.model,
                "window": args.window,
                "token_stride": args.token_stride,
                "windows_done": state["windows_done"],
                "labels": None,   # aucune étiquette : apprentissage non supervisé
                # une tranche de acts.dat n'est CONTINUE que si elle reste dans
                # un seul de ces blocs (voir contiguous_slice dans probe.py)
                "windows": state["windows"],
            })

            done, tot = i + 1, len(windows)
            eta = (time.time() - t_start) / (done - start) * (tot - done)
            print(f"  [{args.split}] {done}/{tot} fenêtres | "
                  f"{state['total_tokens']} vecteurs | "
                  f"{acts_path.stat().st_size / 1e6:.0f} Mo | "
                  f"reste ~{eta / 60:.0f} min", flush=True)

    state_path.unlink(missing_ok=True)
    print(f"[{args.split}] terminé : {state['total_tokens']} vecteurs "
          f"de dimension {d_model} -> {acts_path}")


if __name__ == "__main__":
    main()
