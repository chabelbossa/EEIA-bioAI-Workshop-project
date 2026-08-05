# Embeddings Evo2 — téléchargement

Trop volumineux pour GitHub, ils sont donc hébergés sur Google Drive.

> ### [**Télécharger les embeddings**](https://drive.google.com/drive/folders/1TL1p0ezAXUrGORAoHr7aVtc0YcxE-5ve?usp=drive_link)

[Cliquez ici pour télécharger les données : https://drive.google.com/drive/folders/1TL1p0ezAXUrGORAoHr7aVtc0YcxE-5ve?usp=drive_link](https://drive.google.com/drive/folders/1TL1p0ezAXUrGORAoHr7aVtc0YcxE-5ve?usp=drive_link)

Placez les trois fichiers dans ce dossier :

```
2-data/embeddings/
├── train.npz
├── val.npz
└── test.npz
```

Pré-extraits avec Evo2-7b, afin que l'atelier ne dépende pas d'un accès à l'API
en direct. Utilisés à partir du **notebook 02**.

## Chargement

```python
import numpy as np

d = np.load("2-data/embeddings/train.npz", allow_pickle=True)
X, y = d["embeddings_blocks.26"], d["labels"]
```

`allow_pickle=True` est obligatoire — `ids` est un tableau d'objets.

| Clé | Dimensions | Description |
|---|---|---|
| `embeddings_blocks.26` | (n, 4096) | activations de la couche 26 |
| `embeddings_blocks.31` | (n, 4096) | activations de la couche 31 |
| `labels` | (n,) | 1 = codant, 0 = non codant |
| `ids` | (n,) | identifiants des fenêtres |

## Organisateurs

```bash
export NVIDIA_API_KEY=nvapi-...
python extract_evo2_embeddings.py \
    --processed_dir ./processed --out_dir ./embeddings \
    --target_layers blocks.26 blocks.31
```
