# Détection de régions codantes dans des génomes bactériens

**Evo2 · représentations biologiques · distillation out-of-fold · modèle compact**

[![CI](https://github.com/chabelbossa/EEIA-bioAI-Workshop-project/actions/workflows/ci.yml/badge.svg)](https://github.com/chabelbossa/EEIA-bioAI-Workshop-project/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-models-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-release%20candidate-orange)](docs/PUBLICATION_CHECKLIST.md)

> Projet collectif réalisé pendant la quatrième semaine de l'**École d'Été sur
> l'Intelligence Artificielle — EEIA 2026**, puis prolongé par des expériences
> d'audit des fuites, de distillation plus rigoureuse et de reproductibilité.

## Problème réellement traité

Le dépôt résout une tâche de **classification binaire de fenêtres d'ADN de 200 nucléotides** : déterminer si une fenêtre provient d'une région **codante** (`label = 1`) ou **non codante** (`label = 0`) d'un génome microbien.

Il ne s'agit pas d'une classification des fonctions de protéines. Cette distinction corrige le titre imprécis du poster final tout en conservant le pipeline effectivement implémenté.

## Question expérimentale

> Peut-on transférer une partie de l'information contenue dans les embeddings Evo2 vers un petit modèle autonome, avec seulement 4 000 exemples, sans lui fournir des cibles teacher issues de la mémorisation ?

## Pipeline principal

```text
Pendant l'entraînement

Séquence ADN
  ├── embedding Evo2 4096-D → teacher MLP → logit out-of-fold ─────┐
  │                                                               │
  └── 761 caractéristiques codon/phase → student MLP ← vrai label ┘

À l'inférence

Séquence ADN
  → 761 caractéristiques biologiques
  → student MLP de 48 961 paramètres
  → probabilité codant / non codant
```

Les logits de distillation sont générés avec `GroupKFold`, en groupant par organisme. Chaque cible teacher provient donc d'un teacher qui n'a vu ni l'exemple ni son organisme.

## Résultats actuellement défendables

### Parcours limité aux exemples disposant d'embeddings Evo2

| Modèle | Accuracy | F1 | ROC-AUC | Paramètres entraînables | Statut |
|---|---:|---:|---:|---:|---|
| Régression logistique sur 4-mers | 78,85 % | 80,80 % | — | 257 | baseline exécutée |
| CNN one-hot | 83,33 % | 84,13 % | — | 10 465 | baseline exécutée |
| Evo2 gelé + tête MLP | 95,80 % | 95,85 % | — | 524 545 | teacher exécuté |
| Student 4-mers distillé, version initiale | ≈ 84,00 % | ≈ 84,31 % | — | 16 513 | parcours pédagogique |
| Même student codon/phase **sans KD** | 89,80 % ± 0,49 | 90,18 % | 95,30 % | 48 961 | moyenne, 3 seeds |
| Student codon/phase **avec KD OOF** | **90,13 % ± 0,17** | **90,52 %** | **96,29 %** | **48 961** | moyenne, 3 seeds |

À architecture et caractéristiques identiques, la KD apporte en moyenne :

- `+0,33` point d'accuracy ;
- `+0,34` point de F1 ;
- `+0,99` point de ROC-AUC ;
- une variance plus faible entre les trois seeds.

Ces chiffres sont des résultats de validation, pas de test final. Deux seeds KD dépassent 90 %, la troisième atteint 89,9 %. La formulation correcte est donc **90,13 % en moyenne**.

### Axe supervisé séparé utilisant toutes les données

| Modèle | Surface d'évaluation | Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|
| ExtraTrees + caractéristiques codon/phase | 9 501 fenêtres canoniques inédites | **95,47 %** | **95,86 %** | **98,57 %** |

Cet axe n'est pas directement comparable au défi limité aux 4 000 embeddings. Il met surtout en évidence la force du feature engineering biologique.

## Contrat de caractéristiques public

Le package `bioai` expose un vecteur versionné `codon-phase-761-v1` :

| Bloc | Dimensions |
|---|---:|
| k-mers globaux, k=1..4 | 340 |
| codons dans trois phases, brin direct | 192 |
| codons dans trois phases, reverse-complement | 192 |
| résumés start/stop/segments sans stop | 24 |
| fréquences A/C/G/T par phase | 12 |
| dispersion GC entre phases | 1 |
| **Total** | **761** |

Un test compare l'implémentation publique au module historique afin de détecter toute dérive silencieuse dans l'ordre ou le calcul des features.

## Installation

Python 3.11 est recommandé.

```bash
git clone https://github.com/chabelbossa/EEIA-bioAI-Workshop-project.git
cd EEIA-bioAI-Workshop-project

python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1      # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,dev]"
pre-commit install
```

Vérification :

```bash
make lint
make test
make smoke
```

Les tests et la CI n'ont besoin ni des embeddings Evo2 ni d'un GPU.

## Utilisation du package

### Inspecter le contrat

```bash
bioai about
```

### Extraire les 761 caractéristiques

```bash
bioai features \
  --sequence ATGATGATGATG \
  --output features.npy
```

### Auditer les chevauchements exacts et reverse-complements

```bash
bioai audit \
  --train 2-data/processed/train.csv \
  --evaluation 2-data/processed/val.csv \
  --output reports/overlap-audit.json
```

### Évaluer un checkpoint sur deux surfaces

```bash
bioai evaluate \
  --checkpoint artifacts/student-oof.pt \
  --train 2-data/processed/train.csv \
  --evaluation 2-data/processed/val.csv \
  --output reports/student-validation.json
```

Le rapport contient les métriques sur :

1. toutes les lignes ;
2. uniquement les lignes sans équivalent exact ni reverse-complement dans le train.

Cette commande permettra de produire proprement le résultat encore manquant sur les **993 fenêtres canoniques inédites** du sous-ensemble Evo2 dès qu'un checkpoint final versionné sera choisi.

### Prédire une séquence

```bash
bioai predict \
  --checkpoint artifacts/student-oof.pt \
  --sequence ATGCGT... \
  --device cpu
```

## API optionnelle

```bash
python -m pip install -e ".[serve]"
export BIOAI_CHECKPOINT=/chemin/student-oof.pt
uvicorn bioai.api:app --host 0.0.0.0 --port 8000
```

Voir [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Le service est une démonstration de recherche, sans authentification ni rate limiting.

## Données et embeddings Evo2

Les CSV traités sont présents sous `2-data/processed/`. Les embeddings Evo2 sont trop volumineux pour GitHub et se téléchargent séparément selon :

[`2-data/embeddings/README.md`](2-data/embeddings/README.md)

```text
2-data/embeddings/
├── train.npz
├── val.npz
└── test.npz
```

La provenance, les licences et les checksums des FASTA, GFF et embeddings restent à compléter avant une release archivale. Voir la [data card](docs/DATA_CARD.md).

## Structure

```text
.
├── src/bioai/              # cœur réutilisable, CLI, audit, modèle et évaluation
├── tests/                  # tests déterministes sans gros artefacts
├── configs/                # configuration reconstruite du student OOF
├── docs/                   # méthode, résultats, cards, limites, publication
├── 1-Introduction/         # matériel pédagogique initial
├── 2-data/                 # construction des données et embeddings externes
├── day1/                   # baselines
├── day2/                   # teacher Evo2
├── day3/                   # distillation et expériences avancées
├── day4/                   # compromis précision / taille / latence
├── day5/                   # piste SAE exploratoire
└── HISTORIQUE_DETAILLE_DU_PROJET.md
```

Notebooks à lire en priorité :

1. [`day1/01_kmer_and_cnn_baselines.ipynb`](day1/01_kmer_and_cnn_baselines.ipynb) ;
2. [`day2/02_evo2_embeddings_and_classifier.ipynb`](day2/02_evo2_embeddings_and_classifier.ipynb) ;
3. [`day3/03_serious_oof_distillation.ipynb`](day3/03_serious_oof_distillation.ipynb) ;
4. [`day4/04_compression_analysis_and_wrapup.ipynb`](day4/04_compression_analysis_and_wrapup.ipynb).

## Limites principales

- Le gain KD de `0,33` point repose actuellement sur trois seeds ; sa significativité statistique n'est pas établie.
- Le niveau global de 90,13 % dépend fortement des 761 caractéristiques ; la KD seule n'explique pas le passage de 84 % à 90 %.
- Sept validations sur 1 000 ont un équivalent exact ou reverse-complement dans le train.
- La métrique du student final sur les 993 validations inédites reste à exécuter.
- Le split test reste fermé jusqu'au gel collectif d'un seul pipeline.
- La piste d'autoencodeur parcimonieux n'est pas un résultat validé.
- Le checkpoint final n'est pas encore attaché à une release.

## Documentation

- [Méthodologie](docs/METHODOLOGY.md)
- [Résultats et niveaux de preuve](docs/RESULTS.md)
- [Data card](docs/DATA_CARD.md)
- [Model card](docs/MODEL_CARD.md)
- [Limites](docs/LIMITATIONS.md)
- [Reproductibilité](docs/REPRODUCIBILITY.md)
- [Déploiement](docs/DEPLOYMENT.md)
- [Checklist de publication](docs/PUBLICATION_CHECKLIST.md)
- [Historique détaillé](HISTORIQUE_DETAILLE_DU_PROJET.md)
- [Crédits et participants](CONTRIBUTORS.md)

## Crédits et attribution

Le matériel pédagogique initial a été conçu par **Généreux Akotenou** dans le contexte de l'EEIA 2026. Le projet final a été réalisé par huit participants avec trois superviseurs, puis prolongé par des expérimentations et une documentation supplémentaires.

Les contributions individuelles ne doivent pas être déduites de l'ordre des noms. Elles seront confirmées dans [`CONTRIBUTORS.md`](CONTRIBUTORS.md) avant `v1.0`. Le fichier `CITATION.cff.template` ne doit devenir `CITATION.cff` qu'après validation collective de l'authorship.

## Formulation rigoureuse à retenir

> Nous avons construit un student MLP de 48 961 paramètres utilisant 761 caractéristiques biologiques calculées depuis l'ADN. Avec 4 000 exemples et une distillation out-of-fold groupée par organisme, il atteint 90,13 % d'accuracy moyenne sur trois seeds et ne dépend plus d'Evo2 à l'inférence. La comparaison avec le même student sans distillation mesure un gain moyen de 0,33 point d'accuracy et de 0,99 point de ROC-AUC.

## Statut de publication

Le dépôt est une **release candidate technique**, pas encore `v1.0`. Les blocages restants sont suivis dans [`docs/PUBLICATION_CHECKLIST.md`](docs/PUBLICATION_CHECKLIST.md) : attribution, provenance juridique, checksums, réexécution des notebooks, checkpoint final et unique ouverture du test.

## Licence

Le code est distribué sous licence [MIT](LICENSE). Les données biologiques, annotations, embeddings, logos, photographies et poster peuvent relever d'autres droits et conditions.
