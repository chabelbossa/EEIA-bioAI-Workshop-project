# Méthodologie

Ce document décrit le protocole actuellement utilisé pour distinguer des fenêtres
d’ADN bactérien **codantes** et **non codantes**. Il sépare le parcours
pédagogique d’origine des expériences avancées ajoutées ensuite.

## 1. Tâche

La cible est binaire :

- `1` : fenêtre extraite d’une région annotée `CDS` ;
- `0` : fenêtre extraite d’une région intergénique, hors des `CDS`.

Chaque observation contient notamment :

```text
id, split, organism, seqid, start, end, strand, label, sequence
```

La longueur nominale d’une fenêtre est de **200 nucléotides**.

## 2. Construction des données

Les sources brutes associent :

- un fichier **FASTA**, contenant la séquence du génome ;
- un fichier **GFF**, contenant les annotations et les coordonnées des `CDS`.

Le script [`2-data/build_dataset.py`](../2-data/build_dataset.py) :

1. lit les séquences FASTA ;
2. extrait les intervalles `CDS` du GFF ;
3. génère les fenêtres positives à l’intérieur des `CDS` ;
4. fusionne les intervalles codants pour déterminer les régions intergéniques ;
5. génère les fenêtres négatives dans ces régions ;
6. équilibre les deux classes par organisme ;
7. écrit un CSV par split.

Avec les valeurs par défaut, `window = 200` et `stride = 200`, donc les fenêtres
successives ne se chevauchent pas au sein d’un même intervalle.

### Orientation des séquences

Les fenêtres positives situées sur le brin `-` sont converties en
reverse-complement. Les fenêtres négatives sont actuellement conservées dans
l’orientation de référence. Cette asymétrie doit être étudiée explicitement :
elle peut être biologiquement raisonnable pour représenter une `CDS` dans son sens
de lecture, mais elle peut également créer un signal de raccourci entre les
classes.

Les contrôles recommandés avant `v1.0` sont :

- entraîner et évaluer avec canonicalisation systématique ;
- appliquer une augmentation aléatoire par reverse-complement aux deux classes ;
- mesurer l’invariance des prédictions entre `seq` et `RC(seq)` ;
- comparer les performances avec et sans information d’orientation.

## 3. Splits disponibles

| Split | Lignes brutes | Usage actuel |
|---|---:|---|
| Train | 58 552 | apprentissage et expériences internes |
| Validation | 10 374 | comparaison des pipelines après sélection interne |
| Test | 18 448 | évaluation finale, encore fermée |

Les organismes du train et de la validation sont disjoints dans le dataset
courant :

- train : 20 organismes ;
- validation : 4 organismes.

Trois séquences ne possèdent pas exactement 200 nucléotides : deux dans le train
et une dans la validation. Les expériences avancées qui exigent une longueur fixe
utilisent donc 58 550 lignes train et 10 373 lignes validation.

## 4. Sous-ensemble Evo2

Les embeddings Evo2 précalculés existent pour :

- 4 000 exemples train équilibrés ;
- 1 000 exemples validation équilibrés ;
- un fichier test conservé hors du protocole de sélection.

Les exemples sont alignés avec les CSV via leur colonne `id`. Un `assert` vérifie
que l’ordre et les labels correspondent.

Le défi principal de distillation reste volontairement limité à ces 4 000
exemples afin de comparer les représentations sur une quantité de données
constante.

## 5. Représentations et modèles

### 5.1. Baseline 4-mers

Une séquence est représentée par les fréquences normalisées de ses 4-mers :

```text
4^4 = 256 caractéristiques
```

Une régression logistique et un arbre de décision servent de références légères.

### 5.2. CNN one-hot

Chaque nucléotide est encodé sur quatre canaux `A/C/G/T`. L’entrée du CNN est donc
un tenseur de forme `(4, 200)`. Deux convolutions 1D et une tête linéaire apprennent
les motifs locaux.

### 5.3. Teacher Evo2

Evo2 reste gelé. Pour chaque séquence, une activation de dimension 4 096 est
chargée depuis les fichiers `.npz`. Une petite tête MLP est entraînée :

```text
4096 → Linear(4096, 128) → ReLU → Dropout(0.1) → Linear(128, 1)
```

Cette tête possède 524 545 paramètres entraînables.

### 5.4. Student initial

Le student pédagogique d’origine reçoit les 256 fréquences de 4-mers et utilise un
petit MLP. La perte combine labels durs et probabilités teacher :

```text
L = alpha × L_hard + (1 - alpha) × T² × L_soft
```

Cette première approche a montré les limites d’une représentation trop faible et
de cibles teacher calculées sur des exemples vus pendant l’entraînement.

### 5.5. Représentation codon/phase

Le fichier
[`day3/src/advanced_features.py`](../day3/src/advanced_features.py) construit
**761 caractéristiques** :

| Bloc | Dimensions |
|---|---:|
| k-mers globaux de longueurs 1 à 4 | 340 |
| codons, 3 phases, brin direct | 192 |
| codons, 3 phases, reverse-complement | 192 |
| résumés start/stop/segments sans stop | 24 |
| fréquences A/C/G/T par phase | 12 |
| écart GC entre phases | 1 |
| **Total** | **761** |

### 5.6. Student final

```text
761 entrées
  → Linear(761, 64)
  → LayerNorm(64)
  → GELU
  → Dropout(0.2)
  → Linear(64, 1)
```

Nombre de paramètres :

```text
761 × 64 + 64   = 48 768
LayerNorm       =    128
64 × 1 + 1      =     65
Total           = 48 961
```

## 6. Distillation out-of-fold groupée

Le notebook de référence est
[`day3/03_serious_oof_distillation.ipynb`](../day3/03_serious_oof_distillation.ipynb).

### 6.1. Problème résolu

Une cible teacher ne doit pas provenir d’un modèle ayant appris sur le même
exemple. Sinon, le student peut imiter de la mémorisation plutôt qu’une connaissance
généralisable.

### 6.2. Production des cibles OOF

`GroupKFold(n_splits=5)` utilise `organism` comme groupe :

1. quatre organismes sont retenus dans le fold OOF ;
2. un teacher neuf est entraîné sur les seize organismes restants ;
3. il prédit les exemples des organismes retenus ;
4. l’opération est répétée cinq fois ;
5. chaque exemple train reçoit exactement un logit OOF.

### 6.3. Sélection interne

Avant de produire les logits OOF définitifs sur les 4 000 exemples, un holdout
interne groupé sépare :

| Sous-ensemble | Exemples | Organismes | Usage |
|---|---:|---:|---|
| Développement | 3 246 | 16 | entraîner les candidates et produire les cibles OOF |
| Réglage interne | 754 | 4 | choisir `alpha`, `T`, époque et seuil |

Configuration choisie :

```text
alpha = 0.5
T = 1
57 époques
seuil de décision = 0.265
```

La validation officielle de 1 000 exemples n’est pas utilisée pour choisir ces
valeurs.

### 6.4. Évaluation multi-seeds

Le même student est ensuite entraîné :

- avec les labels durs uniquement ;
- avec la perte hybride de distillation OOF.

Les seeds actuelles sont `7`, `42` et `123`. Les métriques moyennes permettent
d’isoler le gain de la KD pour une architecture et des features identiques.

## 7. Audit des chevauchements

Deux séquences sont considérées équivalentes si elles sont :

- exactement identiques ;
- identiques après reverse-complement.

La fonction `canonical_sequence` retient la valeur lexicographiquement minimale
entre une séquence et son reverse-complement.

### Validation complète

L’audit de la validation complète a identifié :

- 848 séquences uniques exactes communes aux splits ;
- 863 séquences uniques communes après canonicalisation ;
- 872 lignes de validation concernées.

Les résultats prudents du modèle ExtraTrees sont donc calculés sur les 9 501
fenêtres canoniques inédites.

### Sous-ensemble Evo2

Sur les 1 000 validations disposant d’embeddings, 7 possèdent un équivalent exact
ou reverse-complement dans les 4 000 exemples train. Le score séparé du student
final sur les 993 validations inédites constitue une action obligatoire avant la
release finale.

## 8. Mesures d’efficacité

Le fichier [`day3/src/benchmark.py`](../day3/src/benchmark.py) mesure :

- le nombre de paramètres entraînables ;
- la taille sérialisée du `state_dict` ;
- la médiane de la latence d’extraction des features ;
- la médiane du forward PyTorch ;
- la synchronisation CUDA ou MPS lorsque nécessaire.

Les mesures actuellement rapportées sont locales. Elles doivent toujours être
accompagnées du CPU/GPU, du système, des versions de bibliothèques, du nombre de
répétitions et de la taille du batch.

## 9. Split test

Le split test doit rester fermé jusqu’à ce que :

1. un pipeline final unique soit choisi ;
2. toutes les transformations soient gelées ;
3. le seuil soit gelé ;
4. le protocole soit réexécutable ;
5. la release candidate soit revue.

Après ouverture, aucune nouvelle optimisation ne devra être effectuée sur la base
du score test. Toute modification ultérieure nécessitera un nouveau jeu de test.
