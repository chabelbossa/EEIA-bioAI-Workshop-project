#!/usr/bin/env bash
# EEIA — bioAI Workshop — activations Evo2 pour la piste bonus (Jour 5).
# Organisateur uniquement.
#
#   export NVIDIA_API_KEY=nvapi-...
#   bash autoencoder.embedding.sh
#
# Longues fenêtres, aucune étiquette, activations gardées par token.
# Reprenable : si ça s'interrompt, relancez la même commande.
set -euo pipefail

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "Erreur : NVIDIA_API_KEY n'est pas défini." >&2
    echo "  export NVIDIA_API_KEY=nvapi-..." >&2
    exit 1
fi

# TOKEN_STRIDE=1 : on garde TOUTES les positions.
# Indispensable — l'analyse de périodicité du Jour 5 fait une FFT sur des positions
# CONSÉCUTIVES. Avec un pas de 8, un signal de période 3 (le cadre de lecture des
# codons) serait replié et apparaîtrait vers 1,5 : le résultat serait faux.
#
# Une fenêtre de 4096 pb ≈ 40 s d'API et ≈ 67 Mo sur disque (4096 vecteurs).
#
#   20 fenêtres train ≈ 14 min  ->  81 920 vecteurs, ~1,3 Go
#    5 fenêtres eval  ≈  4 min  ->  20 480 vecteurs, ~335 Mo
#
# Test rapide du pipeline : MAX_TRAIN=1 MAX_EVAL=1 bash autoencoder.embedding.sh
WINDOW=${WINDOW:-4096}
TOKEN_STRIDE=${TOKEN_STRIDE:-1}
MAX_TRAIN=${MAX_TRAIN:-20}
MAX_EVAL=${MAX_EVAL:-5}

for SPLIT in train eval; do
    if [ "$SPLIT" = "train" ]; then MAX=$MAX_TRAIN; SRC=./raw/train; else MAX=$MAX_EVAL; SRC=./raw/val; fi
    [ -d "$SRC" ] || SRC=./raw

    echo "=== $SPLIT ($MAX fenêtres de ${WINDOW} pb, depuis $SRC)"
    python extract_autoencoder_activations.py \
        --raw_dir "$SRC" \
        --out_dir ./autoencoder \
        --split "$SPLIT" \
        --window "$WINDOW" \
        --token_stride "$TOKEN_STRIDE" \
        --max_windows "$MAX" \
        --layer blocks.26
done

echo
echo "Terminé. Vérification :"
python - <<'PY'
import json, pathlib
for split in ("train", "eval"):
    m = pathlib.Path(f"./autoencoder/{split}/meta.json")
    if m.exists():
        d = json.loads(m.read_text())
        print(f"  {split}: {d['total_tokens']} vecteurs × {d['d_model']} dims "
              f"({d['dtype']}, couche {d['layer']})")
PY
