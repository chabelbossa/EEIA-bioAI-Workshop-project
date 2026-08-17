# Détection de régions codantes dans des génomes bactériens

**Evo2 · représentations biologiques · distillation out-of-fold · modèles compacts**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-models-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-release%20candidate-orange)](HISTORIQUE_DETAILLE_DU_PROJET.md)

> Projet collectif réalisé pendant la quatrième semaine de l’**École d’Été sur
> l’Intelligence Artificielle — EEIA 2026**, puis prolongé par des expériences
> de validation, d’audit des fuites et de distillation plus rigoureuses.

## Résumé

Le projet traite une tâche de **classification binaire de fenêtres d’ADN de
200 nucléotides** : prédire si une fenêtre provient d’une région **codante**
(`label = 1`) ou **non codante** (`label = 0`) d’un génome bactérien.

Nous comparons trois familles d’approches :

1. des baselines légères utilisant des fréquences de k-mers ;
2. un CNN 1D entraîné directement sur les nucléotides encodés en one-hot ;
3. un teacher fondé sur des embeddings **Evo2**, puis un student autonome
   entraîné par **distillation de connaissances out-of-fold**, groupée par
   organisme.

Le résultat central n’est pas de « compresser tout Evo2 ». Il montre qu’un
petit MLP peut exploiter des cibles teacher produites hors échantillon tout en
utilisant, à l’inférence, uniquement des caractéristiques calculées directement
sur l’ADN.

## Question expérimentale principale

> Peut-on transférer une partie de l’information contenue dans les embeddings
> Evo2 vers un petit modèle autonome, avec seulement 4 000 exemples, sans lui
> fournir des cibles teacher issues de la mémorisation ?

## Pipeline principal

```text
Pendant l’entraînement

Séquence ADN
  ├── embedding Evo2 4096-D → teacher MLP → logit out-of-fold ─────┐
  │                                                               │
  └── 761 caractéristiques codon/phase → student MLP ← vrai label ┘

À l’inférence

Séquence ADN
  → 761 caractéristiques biologiques
  → student MLP de 48 961 paramètres
  → probabilité codant / non codant
```

Les 761 caractéristiques comprennent notamment :

- les fréquences globales des k-mers de longueurs 1 à 4 ;
- les fréquences de codons dans les trois phases de lecture ;
- les mêmes informations sur le reverse-complement ;
- des résumés liés aux codons start/stop et aux segments sans stop ;
- la composition en bases selon la phase et une mesure de périodicité GC.

## Résultats actuellement défendables

### Parcours limité aux 4 000 exemples disposant d’embeddings Evo2

| Modèle | Accuracy | F1 | AUC | Paramètres entraînables | Statut |
|---|---:|---:|---:|---:|---|
| Régression logistique sur 4-mers | 78,85 % | 80,80 % | — | 257 | baseline exécutée |
| CNN one-hot | 83,33 % | 84,13 % | — | 10 465 | baseline exécutée |
| Evo2 gelé + tête MLP | 95,80 % | 95,85 % | — | 524 545 | teacher exécuté |
| Student 4-mers distillé, version initiale | ≈ 84,00 % | ≈ 84,31 % | — | 16 513 | parcours pédagogique |
| Même student codon/phase **sans KD** | 89,80 % ± 0,49 | 90,18 % | 95,30 % | 48 961 | moyenne sur 3 seeds |
| Student codon/phase **avec KD OOF** | **90,13 % ± 0,17** | **90,52 %** | **96,29 %** | **48 961** | moyenne sur 3 seeds |

Dans cette comparaison contrôlée, la distillation apporte en moyenne :

- `+0,33` point d’accuracy ;
- `+0,34` point de F1 ;
- `+0,99` point d’AUC ;
- une variance plus faible entre les trois seeds testées.

Ces résultats sont rapportés sur les 1 000 exemples de validation officiels. Un
audit a identifié 7 fenêtres possédant un équivalent exact ou
reverse-complement dans le train ; l’évaluation séparée sur les 993 fenêtres
canoniques inédites reste à ajouter avant la release `v1.0`.

### Axe supervisé séparé utilisant toutes les données

| Modèle | Surface d’évaluation | Accuracy | F1 | AUC |
|---|---|---:|---:|---:|
| ExtraTrees + caractéristiques codon/phase | 9 501 fenêtres canoniques inédites | **95,47 %** | **95,86 %** | **98,57 %** |

Cet axe n’est pas directement comparable au défi de distillation limité à 4 000
exemples. Il montre surtout la force du feature engineering biologique sur cette
tâche.

## Protocole anti-fuite

L’expérience finale de distillation suit les règles suivantes :

- séparation des données par organisme ;
- production des logits teacher avec `GroupKFold` ;
- chaque logit OOF provient d’un teacher qui n’a vu ni l’exemple, ni son
  organisme ;
- réglage interne réalisé sur quatre organismes du train ;
- validation officielle laissée fermée pendant la sélection ;
- split `test` non utilisé pour choisir les modèles ou produire les résultats
  rapportés ici ;
- audit des séquences identiques et des reverse-complements entre les splits.

Le protocole complet est documenté dans
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Limites importantes

- Le gain moyen de `0,33` point attribuable à la KD repose actuellement sur trois
  seeds : il ne doit pas être présenté comme une preuve statistique définitive.
- Le niveau global de 90,13 % dépend fortement des 761 caractéristiques ; la
  distillation seule n’explique pas le passage du student initial à ce niveau.
- Les métriques du student final sur les 993 validations canoniques inédites
  doivent encore être calculées.
- Le split test reste fermé jusqu’au gel définitif d’un pipeline unique.
- Les expériences d’autoencodeur parcimonieux constituent une piste bonus encore
  non validée.
- Les latences dépendent du matériel. Dans la mesure actuelle, l’extraction des
  caractéristiques représente environ 95 % du temps total du student final.

Voir [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) pour le détail.

## Structure du dépôt

```text
.
├── 1-Introduction/        # contexte biologique et matériel pédagogique initial
├── 2-data/                # construction du dataset et chargement des embeddings
├── day1/                  # baselines k-mers et CNN one-hot
├── day2/                  # teacher sur embeddings Evo2
├── day3/                  # distillation et expériences avancées
├── day4/                  # comparaison précision / taille / latence
├── day5/                  # piste SAE exploratoire
├── docs/                  # méthodologie, résultats, limites, reproductibilité
├── HISTORIQUE_DETAILLE_DU_PROJET.md
└── requirements.txt
```

Les notebooks à lire en priorité sont :

1. [`day1/01_kmer_and_cnn_baselines.ipynb`](day1/01_kmer_and_cnn_baselines.ipynb) ;
2. [`day2/02_evo2_embeddings_and_classifier.ipynb`](day2/02_evo2_embeddings_and_classifier.ipynb) ;
3. [`day3/03_serious_oof_distillation.ipynb`](day3/03_serious_oof_distillation.ipynb) ;
4. [`day4/04_compression_analysis_and_wrapup.ipynb`](day4/04_compression_analysis_and_wrapup.ipynb).

## Installation actuelle

Python 3.11 est recommandé.

```bash
git clone https://github.com/chabelbossa/EEIA-bioAI-Workshop-project.git
cd EEIA-bioAI-Workshop-project

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name eeia-bioai \
  --display-name "EEIA bioAI"

jupyter lab
```

> L’environnement n’est pas encore verrouillé par un fichier de lock. Une
> configuration reproductible avec `pyproject.toml`, tests et CI fait partie de
> la préparation de la release `v1.0`.

## Données et embeddings Evo2

Les CSV traités sont présents dans `2-data/processed/`. Les embeddings Evo2 sont
trop volumineux pour GitHub et doivent être téléchargés séparément en suivant :

[`2-data/embeddings/README.md`](2-data/embeddings/README.md)

Ils doivent être placés sous :

```text
2-data/embeddings/
├── train.npz
├── val.npz
└── test.npz
```

Les checksums SHA256 seront ajoutés avant la release finale.

## Documentation

- [Méthodologie](docs/METHODOLOGY.md)
- [Résultats et niveaux de preuve](docs/RESULTS.md)
- [Limites et menaces à la validité](docs/LIMITATIONS.md)
- [Reproductibilité](docs/REPRODUCIBILITY.md)
- [Historique détaillé des expériences](HISTORIQUE_DETAILLE_DU_PROJET.md)
- [Crédits et participants](CONTRIBUTORS.md)

## Crédits et attribution

Le matériel pédagogique initial de l’atelier a été conçu par **Généreux
Akotenou** et publié sous licence MIT dans le dépôt EEIA bioAI Workshop. Ce dépôt
conserve cette attribution et documente le travail réalisé par une équipe de
participants de l’EEIA 2026, ainsi que des prolongements expérimentaux menés après
l’atelier.

La liste des participants, superviseurs et règles d’attribution se trouve dans
[`CONTRIBUTORS.md`](CONTRIBUTORS.md). Les contributions individuelles ne doivent
pas être déduites de l’ordre des noms et seront précisées avant la release `v1.0`.

## Formulation rigoureuse à retenir

> Nous avons construit un student MLP de 48 961 paramètres utilisant 761
> caractéristiques biologiques calculées depuis l’ADN. Avec 4 000 exemples et une
> distillation out-of-fold groupée par organisme, il atteint 90,13 % d’accuracy
> moyenne sur trois seeds et ne dépend plus d’Evo2 à l’inférence. La comparaison
> avec le même student sans distillation mesure un gain moyen de 0,33 point
> d’accuracy et de 0,99 point d’AUC.

## Licence

Le code est distribué sous licence [MIT](LICENSE). Les jeux de données, annotations
biologiques, embeddings Evo2 et ressources externes peuvent être soumis à leurs
propres licences et conditions d’utilisation ; leur provenance doit être vérifiée
avant toute redistribution indépendante.
