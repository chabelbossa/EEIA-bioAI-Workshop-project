# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
NVIDIA-API-KEY : https://build.nvidia.com/arc/evo2-7b-forward/modelcard

prepare_probe_genome.py — ORGANISATEUR. Données de SONDAGE du Jour 5.

Prend UN génome et son GFF, et produit une **piste continue** :

    position 0, 1, 2, ... N   du contig, sans trou
    activations Evo2 pour chacune de ces positions
    étiquette codant / non-codant pour chacune de ces positions

C'est le seul jeu de données du projet où activations et annotations sont liées
position par position. Il ne sert PAS à entraîner : uniquement à répondre à
« cette caractéristique du SAE s'allume-t-elle sur les gènes ? ».

Pourquoi un fichier à part : le dump d'entraînement du SAE
(extract_autoencoder_activations.py) est une concaténation de fenêtres tirées au
hasard, sans coordonnées. On ne peut donc pas y rattacher un GFF de façon fiable.
Ici, la région est contiguë et ses coordonnées sont écrites dans meta.json.

La région demandée est découpée en morceaux de --chunk pb envoyés à l'API dans
l'ordre, puis recollés : la piste finale est continue.

Contrôle d'annotation
---------------------
Une bactérie est codante à ~85-90 %. Si le GFF fourni couvre beaucoup moins, il est
incomplet, et les zones « non-codantes » seraient en fait des gènes non annotés :
le script REFUSE de continuer (--allow_sparse_gff pour passer outre).

Sortie
------
    <out_dir>/<nom>/acts.dat     (longueur, d_model) float32
    <out_dir>/<nom>/labels.npy   (longueur,) int8 — 1 = CDS, 0 = intergénique
    <out_dir>/<nom>/sequence.txt la séquence, pour vérification
    <out_dir>/<nom>/meta.json    contig, start, end, morceaux, statistiques

Usage
-----
export NVIDIA_API_KEY=nvapi-...
python prepare_probe_genome.py \\
    --fasta ./raw/train/GCA_000240015.1_ASM24001v1.fasta \\
    --gff   ./raw/train/GCA_000240015.1_ASM24001v1.gff \\
    --length 32768 --chunk 8192

Reprenable : relancez la même commande, elle continue au morceau suivant.
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

FORWARD_URL_TMPL = "https://health.api.nvidia.com/v1/biology/arc/{model}/forward"
MAX_RETRIES = 10
VALID_BASES = set("ACGT")


# --------------------------------------------------------------------- lecture
def read_fasta(path):
    seqs, name, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.upper())
    if name:
        seqs[name] = "".join(chunks)
    return seqs


def read_cds(path):
    """{contig: [(start0, end), ...]} — CDS en coordonnées 0-based, brins confondus."""
    spans = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            spans.setdefault(f[0], []).append((int(f[3]) - 1, int(f[4])))
    return spans


def coding_mask(cds_spans, contig, start, end):
    mask = np.zeros(end - start, dtype=np.int8)
    for s, e in cds_spans.get(contig, []):
        if e <= start or s >= end:
            continue
        mask[max(s, start) - start:min(e, end) - start] = 1
    return mask


def check_annotation(seqs, cds, min_coverage, allow_sparse):
    """Un GFF tronqué produirait un masque faux : on le détecte avant de payer l'API."""
    total = sum(len(s) for s in seqs.values())
    n_cds = sum(len(v) for v in cds.values())
    covered = sum(int((np.array(v)[:, 1] - np.array(v)[:, 0]).sum())
                  for v in cds.values() if v)
    cov = covered / total if total else 0.0
    print(f"  annotation : {n_cds} CDS, {cov:.1%} du génome couvert")
    if cov < min_coverage:
        msg = (f"GFF suspect : {cov:.1%} de couverture seulement. Une bactérie est "
               f"codante à ~85-90 % — ce fichier est probablement tronqué, et les "
               f"positions « non-codantes » seraient en fait des gènes non annotés.\n"
               f"  Utilisez un génome complètement annoté (dossier raw/train/), "
               f"ou --allow_sparse_gff si vous savez ce que vous faites.")
        if not allow_sparse:
            raise SystemExit("ERREUR : " + msg)
        print("  ATTENTION : " + msg)
    return cov, n_cds


def pick_span(seqs, cds, length, contig=None, start=None):
    """Choisit une région contiguë riche en alternances gène / intergène."""
    if contig is None:
        contig = max(seqs, key=lambda c: len(seqs[c]))
    seq = seqs[contig]
    if len(seq) < length:
        raise SystemExit(f"{contig} fait {len(seq)} pb, moins que --length")

    if start is not None:
        return contig, start

    mask = coding_mask(cds, contig, 0, len(seq)).astype(np.int32)
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    valid = np.isin(arr, np.frombuffer(b"ACGT", dtype=np.uint8)).astype(np.int32)
    trans = np.abs(np.diff(mask))

    c_valid = np.concatenate(([0], np.cumsum(valid)))
    c_trans = np.concatenate(([0], np.cumsum(trans)))
    c_mask = np.concatenate(([0], np.cumsum(mask)))

    step = max(length // 8, 1)
    starts = np.arange(0, len(seq) - length + 1, step)
    ends = starts + length
    clean = (c_valid[ends] - c_valid[starts]) == length
    frac = (c_mask[ends] - c_mask[starts]) / length
    nt = c_trans[ends - 1] - c_trans[starts]
    ok = clean & (frac >= 0.5) & (frac <= 0.95)     # bactérie : majoritairement codant
    if not ok.any():
        raise SystemExit("aucune région propre trouvée — essayez une autre longueur")
    i = int(np.flatnonzero(ok)[np.argmax(nt[ok])])
    print(f"  région : {contig}:{starts[i]}-{ends[i]} "
          f"({frac[i]:.0%} codantes, {nt[i]} frontières gène/intergène)")
    return contig, int(starts[i])


# --------------------------------------------------------------------- API
def get_activations(sequence, layer, api_key, model):
    url = FORWARD_URL_TMPL.format(model=model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers,
                              json={"sequence": sequence, "output_layers": [layer]},
                              timeout=900)
        except requests.exceptions.RequestException as exc:
            wait = 30 * (attempt + 1)
            print(f"  [réseau] {type(exc).__name__} — nouvel essai dans {wait}s")
            time.sleep(wait)
            continue
        if r.status_code in (429, 503, 400, 504):
            wait = 60 * (attempt + 1)
            print(f"  [{r.status_code}] nouvel essai dans {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        npz = np.load(io.BytesIO(base64.b64decode(r.json()["data"].encode("ascii"))))
        key = f"{layer}.output" if f"{layer}.output" in npz.files else layer
        acts = npz[key]
        return acts[0] if acts.ndim == 3 else acts
    raise RuntimeError("échec après plusieurs tentatives")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--out_dir", default="./probe")
    ap.add_argument("--name", default=None)
    ap.add_argument("--contig", default=None, help="défaut : le plus long")
    ap.add_argument("--start", type=int, default=None,
                    help="défaut : la région la plus alternée gène/intergène")
    ap.add_argument("--length", type=int, default=32768,
                    help="longueur TOTALE de la piste continue")
    ap.add_argument("--chunk", type=int, default=8192,
                    help="taille des morceaux envoyés à l'API (recollés ensuite)")
    ap.add_argument("--layer", default="blocks.26")
    ap.add_argument("--model", default="evo2-7b")
    ap.add_argument("--min_gff_coverage", type=float, default=0.5)
    ap.add_argument("--allow_sparse_gff", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("Exportez d'abord NVIDIA_API_KEY.")

    name = args.name or Path(args.fasta).stem
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    acts_path, state_path = out / "acts.dat", out / "_state.json"

    print(f"[{name}] lecture de {Path(args.fasta).name} et {Path(args.gff).name}")
    seqs, cds = read_fasta(args.fasta), read_cds(args.gff)
    cov, n_cds = check_annotation(seqs, cds, args.min_gff_coverage, args.allow_sparse_gff)
    contig, start = pick_span(seqs, cds, args.length, args.contig, args.start)

    end = start + args.length
    seq = seqs[contig][start:end]
    mask = coding_mask(cds, contig, start, end)

    n_chunks = (args.length + args.chunk - 1) // args.chunk
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": 0}
    if state["done"] and state.get("span") != [contig, start, end]:
        print("  région différente de la reprise en cours — on repart de zéro")
        state = {"done": 0}
    if state["done"] == 0 and acts_path.exists():
        acts_path.unlink()
    if state["done"]:
        print(f"  reprise : {state['done']}/{n_chunks} morceaux déjà extraits")

    d_model = state.get("d_model")
    t0 = time.time()
    with open(acts_path, "ab") as fh:
        for c in range(state["done"], n_chunks):
            a, b = c * args.chunk, min((c + 1) * args.chunk, args.length)
            acts = np.ascontiguousarray(
                get_activations(seq[a:b], args.layer, api_key, args.model),
                dtype=np.float32)
            if acts.shape[0] != b - a:
                raise RuntimeError(f"l'API a renvoyé {acts.shape[0]} positions "
                                   f"pour {b - a} demandées — alignement impossible")
            d_model = d_model or int(acts.shape[1])
            fh.write(acts.tobytes()); fh.flush()
            state.update({"done": c + 1, "d_model": d_model,
                          "span": [contig, start, end]})
            state_path.write_text(json.dumps(state))
            eta = (time.time() - t0) / (c + 1 - 0) * (n_chunks - c - 1)
            print(f"  morceau {c + 1}/{n_chunks} ({a}-{b}) | "
                  f"reste ~{eta / 60:.0f} min", flush=True)

    # étiquettes et métadonnées : écrites APRÈS, sur toute la piste
    np.save(out / "labels.npy", mask)
    (out / "sequence.txt").write_text(seq)

    runs_cds = np.diff(np.flatnonzero(np.diff(np.r_[0, mask, 0])))[::2]
    (out / "meta.json").write_text(json.dumps({
        "total_tokens": int(args.length), "d_model": int(d_model),
        "dtype": "float32", "contig": contig,
        "start": int(start), "end": int(end), "length": int(args.length),
        "chunk": int(args.chunk), "n_chunks": int(n_chunks),
        "continuous": True,
        "coding_fraction": float(mask.mean()),
        "n_cds_in_span": int(len(runs_cds)),
        "gff_coverage_genome": float(cov), "gff_n_cds_genome": int(n_cds),
        "layer": args.layer, "model": args.model,
        "source_fasta": str(args.fasta), "source_gff": str(args.gff),
    }, indent=2))
    state_path.unlink(missing_ok=True)

    size = acts_path.stat().st_size
    assert size == args.length * d_model * 4, (size, args.length, d_model)
    print(f"[{name}] piste continue de {args.length} pb × {d_model} dims "
          f"| {mask.mean():.0%} codantes | {len(runs_cds)} gènes -> {out}")


if __name__ == "__main__":
    main()
