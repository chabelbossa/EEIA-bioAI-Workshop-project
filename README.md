# EEIA: bioAI Workshop · Semaine 4

**De l'ADN aux modèles**: une semaine pour aller d'un génome bactérien brut
jusqu'à un classifieur minuscule distillé depuis un modèle de fondation génomique.

<!-- Auteur : Généreux Akotenou: PhD student, BioinformaticLabs College of Computing / UM6P
· [dépôt GitHub](https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop) -->

---

## 1. Commencez par lire le guide

**Avant d'ouvrir le moindre notebook**, lisez [`1-Introduction/guide.md`](1-Introduction/guide.md).
Il pose le contexte biologique, la tâche à résoudre (codant vs non-codant), le déroulé de
la semaine et ce qui est attendu comme livrable. Les notebooks supposent que vous l'avez lu.

La présentation d'ouverture est dans [`1-Introduction/presentation`](1-Introduction/presentation)
(ouvrez-la simplement dans un navigateur).

## 2. Préparez l'environnement Jupyter

<!-- Toutes les commandes ci-dessous se lancent **depuis ce dossier `week4/`**.
Python **3.10 ou plus récent** est requis. -->

### Option A: `venv` (recommandée, aucune installation supplémentaire)

```bash
# 1. créer l'environnement
python3 -m venv .venv

# 2. l'activer
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)

# 3. installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. enregistrer l'environnement comme noyau Jupyter
python -m ipykernel install --user --name eeia-bioai --display-name "(eeia-bioai)"
```

### Option B: `conda`

```bash
conda create -n eeia-bioai python=3.11 -y
conda activate eeia-bioai
pip install -r requirements.txt
python -m ipykernel install --user --name eeia-bioai --display-name "(eeia-bioai)"
```

### Lancer Jupyter

```bash
jupyter lab        # ou : jupyter notebook
```

Dans chaque notebook, sélectionnez le noyau **«(eeia-bioai) »**
(menu *Kernel → Change Kernel*). Si vous ne le voyez pas, relancez l'étape
`ipykernel install` puis rechargez la page.

### Vérifier que tout fonctionne

```bash
python -c "import numpy, pandas, sklearn, torch, matplotlib; print('OK', torch.__version__)"
```

### Dépendances

| Paquet | À quoi il sert |
| --- | --- |
| `numpy`, `pandas` | manipulation des séquences et des splits CSV |
| `scikit-learn` | régression logistique, métriques, PCA |
| `torch` | CNN one-hot, tête MLP du teacher, student distillé, SAE |
| `matplotlib` | tous les graphiques |
| `umap-learn` | projection 2D optionnelle des embeddings (`plot_embedding_space(method="umap")`) |
| `requests` | appels à l'API NVIDIA NIM pour Evo2 (Jour 2) |
| `tqdm` | barres de progression lors de l'extraction d'embeddings |
| `jupyter`, `ipykernel` | exécution des notebooks |

> **CPU suffit.** Aucun GPU n'est nécessaire : les embeddings Evo2 sont pré-calculés et
> fournis, et tous les modèles entraînés pendant la semaine sont petits.

## 3. Déroulé de la semaine

| Dossier | Notebook | Contenu |
| --- | --- | --- |
| `1-Introduction/` | `guide.md`, `presentation.html` | contexte, biologie, organisation |
| `2-data/` | `build_dataset.py`, `extract_evo2_embeddings.py` | scripts fournis + données `raw/` et `processed/` |
| `day1/` | `00_biology_intro_and_data_setup.ipynb` | ADN codant vs non-codant, des FASTA/GFF aux fenêtres étiquetées |
| `day1/` | `01_kmer_and_cnn_baselines.ipynb` | k-mers + régression logistique, one-hot + CNN |
| `day2/` | `02_evo2_embeddings_and_classifier.ipynb` | embeddings Evo2 + tête MLP (le « teacher ») |
| `day3/` | `03_knowledge_distillation.ipynb` | distillation du teacher vers un student minuscule |
| `day4/` | `04_compression_analysis_and_wrapup.ipynb` | compromis précision / taille / latence, bilan |
| `day5/` | `bonus_sparse_autoencoder_interpretability.ipynb` | piste bonus : autoencodeurs parcimonieux |

Suivez l'ordre des jours : chaque notebook réutilise les chiffres ou les modèles du
précédent, et le graphique final du Jour 4 rassemble tout.

## 4. Comment travailler dans les notebooks

- Chaque notebook commence par une cellule **« Votre identité »** : double-cliquez dessus
  et complétez nom, binôme et date avant de commencer.
- Les cellules à compléter sont marquées `# TODO :`: remplacez les `...` par votre code.
- Chaque dossier `dayN/` contient un sous-dossier `src/` avec le code d'infrastructure
  déjà écrit (chargement des données, métriques, graphiques) : à utiliser tel quel.
- Un sous-dossier `solution/` accompagne chaque jour avec la version corrigée complète.
  **Essayez d'abord par vous-même**: la solution est là pour se débloquer, pas pour
  démarrer.

## 5. Licence

Code sous licence **MIT** (voir [`LICENSE`](../LICENSE) à la racine du dépôt).
Si vous réutilisez ce matériel pédagogique, merci de citer l'auteur.
