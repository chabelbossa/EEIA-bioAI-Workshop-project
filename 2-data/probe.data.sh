#!/usr/bin/env bash
# EEIA — bioAI Workshop — données de SONDAGE du Jour 5 (organisateur).
#
#   export NVIDIA_API_KEY=nvapi-...
#   bash probe.data.sh
#
# Produit une piste CONTINUE sur un génome de VALIDATION :
#   - le SAE est entraîné sur des fenêtres de raw/train/ -> on sonde ailleurs
#   - les GFF de raw/val/ sont complets (~88 % du génome couvert), contrairement
#     à ceux de raw/test/ qui sont tronqués et donneraient un masque faux
#
# Reprenable : si ça coupe, relancez la même commande.
set -euo pipefail

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "Erreur : NVIDIA_API_KEY n'est pas défini." >&2
    echo "  export NVIDIA_API_KEY=nvapi-..." >&2
    exit 1
fi

# Génome de validation utilisé pour la sonde.
GENOME=${GENOME:-GCA_001890385.1_ASM189038v1}

# LENGTH = longueur totale de la piste, en paires de bases.
# Elle est découpée en morceaux de CHUNK envoyés à l'API dans l'ordre, puis recollés.
#   32768 pb = 4 morceaux ≈ 5 min d'API, ~537 Mo sur disque
#    8192 pb = 1 morceau  ≈ 1,5 min,      ~134 Mo   (test rapide)
LENGTH=${LENGTH:-32768}
CHUNK=${CHUNK:-8192}
NAME=${NAME:-probe-genome}

echo "=== sonde : $GENOME | $LENGTH pb en morceaux de $CHUNK"
python prepare_probe_genome.py \
    --fasta "./raw/val/${GENOME}.fasta" \
    --gff   "./raw/val/${GENOME}.gff" \
    --out_dir ./probe \
    --name "$NAME" \
    --length "$LENGTH" \
    --chunk "$CHUNK" \
    --layer blocks.26

echo
echo "Vérification :"
python - <<PY
import json, pathlib
m = pathlib.Path("./probe/$NAME/meta.json")
if m.exists():
    d = json.loads(m.read_text())
    print(f"  {d['contig']}:{d['start']}-{d['end']}  ({d['total_tokens']} positions "
          f"x {d['d_model']} dims)")
    print(f"  {d['coding_fraction']:.0%} de positions codantes, "
          f"{d['n_cds_in_span']} gènes dans la région")
    print(f"  GFF source : {d['gff_n_cds_genome']} CDS, "
          f"{d['gff_coverage_genome']:.0%} du génome couvert")
PY
