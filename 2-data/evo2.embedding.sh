#!/usr/bin/env bash
# EEIA — bioAI Workshop — extraction des embeddings Evo2 (organisateur uniquement).
#
# La clé API n'est PAS stockée dans ce fichier : exportez-la avant de lancer le script.
#   export NVIDIA_API_KEY=nvapi-...
#   bash evo2.embedding.sh
set -euo pipefail


if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "Erreur : NVIDIA_API_KEY n'est pas défini." >&2
    echo "  export NVIDIA_API_KEY=nvapi-..." >&2
    exit 1
fi

# Nombre de fenêtres par split (échantillonnage équilibré par classe, seed 42).
# Compter ~4 s par fenêtre : 6000 fenêtres ≈ 7 h. Mettre de petites valeurs
# (ex. 2) pour un test rapide du pipeline.
MAX_TRAIN=${MAX_TRAIN:-4000}
MAX_VAL=${MAX_VAL:-1000}
MAX_TEST=${MAX_TEST:-1000}

# Les deux couches sortent du MÊME appel API — la seconde ne coûte aucune requête
# supplémentaire. blocks.26 = couche par défaut des notebooks, blocks.31 = dernière
# couche d'evo2-7b (pour comparer où vit le signal codant/non-codant).
python extract_evo2_embeddings.py \
    --processed_dir ./processed \
    --out_dir ./embeddings \
    --max_per_split "train=${MAX_TRAIN}" "val=${MAX_VAL}" "test=${MAX_TEST}" \
    --target_layers blocks.26 blocks.31
