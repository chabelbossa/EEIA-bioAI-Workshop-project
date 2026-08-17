# Reproductibilité

Ce document décrit :

1. ce qu’une personne externe peut exécuter aujourd’hui ;
2. les conditions nécessaires pour la release `v1.0` ;
3. les artefacts qui doivent accompagner chaque résultat.

## 1. État actuel

### Environnement

Le dépôt fournit actuellement un `requirements.txt` non verrouillé :

```text
numpy
pandas
scikit-learn
torch
matplotlib
umap-learn
requests
tqdm
jupyter
ipykernel
```

Ce fichier facilite l’installation, mais ne garantit pas qu’une exécution future
utilisera les mêmes versions.

### Données

- les FASTA/GFF bruts sont versionnés dans `2-data/raw/` ;
- les CSV traités sont versionnés dans `2-data/processed/` ;
- les embeddings Evo2 sont téléchargés séparément ;
- aucun checksum n’est encore fourni pour les `.npz` ;
- les checkpoints légers sont présents de manière partielle.

### Notebooks

#### Exécutés sans cellule en erreur enregistrée

- `day2/02_evo2_embeddings_and_classifier.ipynb` ;
- `day3/03_knowledge_distillation.ipynb` ;
- `day3/03_serious_oof_distillation.ipynb` ;
- `day4/04_compression_analysis_and_wrapup.ipynb`.

#### Partiellement exécutés

- `day1/00_biology_intro_and_data_setup.ipynb` ;
- `day1/01_kmer_and_cnn_baselines.ipynb` ;
- `day5/bonus_sparse_autoencoder_interpretability.ipynb`.

#### À nettoyer et réexécuter

- `day3/03_codon_phase_advanced_experiments.ipynb` ;
- `day3/03_test_distillation_experiments.ipynb`.

Le second contient en plus des titres/commentaires contradictoires avec ses sorties
actuelles.

## 2. Installation actuelle

Python 3.11 est recommandé.

```bash
git clone https://github.com/chabelbossa/EEIA-bioAI-Workshop-project.git
cd EEIA-bioAI-Workshop-project

python3.11 -m venv .venv
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name eeia-bioai \
  --display-name "EEIA bioAI"

python -c "import numpy, pandas, sklearn, torch, matplotlib; print('OK', torch.__version__)"
jupyter lab
```

Cette procédure est une installation de travail, pas encore le contrat figé de la
release.

## 3. Embeddings Evo2

Suivre [`../2-data/embeddings/README.md`](../2-data/embeddings/README.md), puis
placer :

```text
2-data/embeddings/
├── train.npz
├── val.npz
└── test.npz
```

Avant `v1.0`, la documentation devra indiquer pour chaque fichier :

```text
nom
taille en octets
SHA256
modèle Evo2 exact
couche extraite
version de l’API ou du code d’extraction
date d’extraction
nombre d’exemples
ordre et source des IDs
licence / conditions de redistribution
```

Le fichier test peut être téléchargé, mais il ne doit pas être chargé avant le gel
du pipeline final.

## 4. Ordre de reproduction actuel

### Parcours pédagogique

1. `day1/00_biology_intro_and_data_setup.ipynb` ;
2. `day1/01_kmer_and_cnn_baselines.ipynb` ;
3. `day2/02_evo2_embeddings_and_classifier.ipynb` ;
4. `day3/03_knowledge_distillation.ipynb` ;
5. `day4/04_compression_analysis_and_wrapup.ipynb`.

### Approche finale OOF

1. télécharger les embeddings train/val ;
2. vérifier que les CSV traités sont disponibles ;
3. exécuter `day3/03_serious_oof_distillation.ipynb` ;
4. exécuter `day4/04_compression_analysis_and_wrapup.ipynb`.

Le notebook OOF :

- aligne les embeddings et les CSV par `id` ;
- vérifie les labels ;
- extrait les 761 features ;
- produit les logits teacher OOF par organisme ;
- choisit les hyperparamètres sur un holdout interne ;
- entraîne les variantes hard et KD sur trois seeds ;
- évalue les modèles sur la validation officielle.

## 5. Contrat cible de la release `v1.0`

### 5.1. Environnement

- `pyproject.toml` comme source principale des dépendances ;
- versions minimales et maximales justifiées ;
- fichier de lock généré après exécution propre ;
- Python 3.11 explicitement testé ;
- compatibilité CPU garantie ;
- CUDA/MPS documentés comme options.

### 5.2. Package

La logique critique devra quitter les notebooks pour un package :

```text
src/bioai/
├── data.py
├── audit.py
├── features.py
├── models.py
├── distillation.py
├── evaluation.py
├── benchmark.py
└── cli.py
```

Les notebooks deviendront des clients de ce package au lieu de contenir plusieurs
versions divergentes de la même logique.

### 5.3. Configuration

Les valeurs suivantes devront être sorties du code :

- chemins des données ;
- seeds ;
- nombre de folds ;
- nombre d’époques ;
- learning rates ;
- `alpha` et `T` ;
- seuil ;
- nombre de workers ;
- matériel de benchmark.

Structure proposée :

```text
configs/
├── baseline.yaml
├── teacher.yaml
├── student_hard.yaml
├── student_kd_oof.yaml
└── benchmark.yaml
```

### 5.4. Commandes

Objectif :

```bash
make setup
make verify-data
make test
make train-student
make evaluate
make report
```

Équivalent CLI cible :

```bash
bioai audit-data --config configs/student_kd_oof.yaml
bioai train --config configs/student_kd_oof.yaml
bioai evaluate --checkpoint artifacts/student_kd_oof.pt
bioai predict --sequence ATG...
```

### 5.5. Tests

Tests unitaires minimaux :

- `reverse_complement(reverse_complement(seq)) == seq` ;
- invariants de `canonical_sequence` ;
- indexation déterministe des k-mers ;
- dimension exacte des 761 features ;
- gestion des bases inconnues ;
- absence de NaN/Inf ;
- alignement `id`, label et embedding ;
- domaine valide de `alpha` ;
- forme des logits ;
- calcul du nombre de paramètres ;
- audit des doublons exacts et reverse-complements.

Tests d’intégration :

- construction d’un petit dataset synthétique ;
- entraînement de quelques époques sur CPU ;
- génération de logits OOF couvrant chaque ligne exactement une fois ;
- sérialisation/chargement du student ;
- prédiction CLI.

### 5.6. CI

Une GitHub Action devra exécuter sans gros fichiers :

```text
ruff check
ruff format --check
pytest
smoke test sur données synthétiques
validation des liens Markdown
vérification de l’absence de secrets
```

Les notebooks lourds et embeddings Evo2 ne doivent pas être exigés par la CI de
chaque commit.

## 6. Reproduction des métriques

Chaque table finale doit être générée à partir d’un artefact structuré, par exemple :

```json
{
  "experiment": "student_kd_oof",
  "commit": "<git-sha>",
  "data_manifest": "<sha256>",
  "python": "3.11.x",
  "torch": "x.y.z",
  "device": "cpu|mps|cuda",
  "seeds": [7, 42, 123],
  "metrics": {
    "accuracy_mean": 0.9013,
    "f1_mean": 0.9052,
    "roc_auc_mean": 0.9629
  }
}
```

Aucune valeur ne devrait être recopiée manuellement dans plusieurs notebooks et
documents.

## 7. Manifestes de données

Créer `data/manifest.json` ou `artifacts/data_manifest.json` avec :

- fichiers ;
- SHA256 ;
- nombre de lignes ;
- labels par classe ;
- organismes ;
- longueur minimale/maximale ;
- taux de bases ambiguës ;
- nombre de séquences uniques ;
- nombre de séquences canoniques uniques ;
- provenance et licence.

Le script doit échouer si les checksums ne correspondent pas.

## 8. Reproductibilité du benchmark

Pour chaque mesure de latence :

- commit Git ;
- version Python/PyTorch/NumPy ;
- modèle CPU/GPU ;
- OS ;
- threads CPU ;
- batch size ;
- warmup ;
- répétitions ;
- médiane, p95 et dispersion ;
- feature extraction séparée du forward ;
- synchronisation matériel.

La comparaison au teacher doit préciser si le coût Evo2 est inclus ou si seuls les
embeddings précalculés sont utilisés.

## 9. Checklist avant ouverture du test

- [ ] données et embeddings vérifiés par checksum ;
- [ ] pipeline final unique choisi ;
- [ ] features gelées ;
- [ ] architecture gelée ;
- [ ] seeds/protocole gelés ;
- [ ] seuil gelé ;
- [ ] score sur 993 validations canoniques inédites ajouté ;
- [ ] hard vs KD répété avec incertitude ;
- [ ] notebooks propres ou pipeline CLI validé ;
- [ ] revue des crédits terminée ;
- [ ] commit de release candidate identifié.

## 10. Après ouverture du test

Le résultat test doit être enregistré une seule fois dans un artefact signé par le
commit de la release candidate.

Après cette ouverture :

- ne pas modifier le modèle en réaction au score ;
- documenter tout écart entre validation et test ;
- ne pas appeler le test « validation » ;
- créer une release immuable ;
- conserver le rapport brut et le manifeste.
