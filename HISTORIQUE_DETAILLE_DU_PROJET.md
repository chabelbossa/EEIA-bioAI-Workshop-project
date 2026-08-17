# Historique détaillé du projet EEIA bioAI

> État reconstitué le 13 août 2026 à partir du dépôt local, de l'historique Git,
> du code des notebooks, de leurs sorties enregistrées et des résultats bruts des
> expérimentations. Ce document distingue les résultats confirmés, les essais
> exploratoires et les limites connues.

## 1. Résumé du projet

Le projet cherche à distinguer des fenêtres d'ADN **codantes** (`label = 1`) et
**non codantes** (`label = 0`). La progression pédagogique initiale était :

1. comprendre la construction des données à partir des génomes ;
2. entraîner des modèles de référence simples ;
3. exploiter les embeddings d'un modèle de fondation génomique, Evo2 ;
4. transférer une partie de cette connaissance vers un petit modèle ;
5. comparer précision, taille et latence ;
6. explorer éventuellement l'interprétabilité avec un autoencodeur parcimonieux.

Notre travail a ensuite prolongé le Jour 3 avec plusieurs expériences visant à
dépasser 90 % d'accuracy. Deux axes doivent rester distincts :

- **meilleur modèle supervisé sur toutes les données** : ExtraTrees avec 761
  caractéristiques codon/phase, 95,47 % d'accuracy sur la partie de validation
  sans séquence identique ou reverse-complement présente dans le train ;
- **défi limité aux 4 000 embeddings Evo2** : petit MLP distillé avec des cibles
  teacher out-of-fold, 90,13 % d'accuracy moyenne sur 1 000 validations.

Le modèle final de la seconde approche est un **MLP**, pas un CNN. Le CNN
multi-échelle appartient à une expérimentation séparée.

## 2. Sources et niveaux de preuve

Les conclusions de ce document reposent sur quatre niveaux de preuve :

| Niveau | Signification |
|---|---|
| Résultat exécuté | La métrique figure dans une sortie de notebook ou un artefact JSON produit par une exécution réelle. |
| Résultat audité | Le protocole, les longueurs, les chevauchements et l'usage du test ont également été contrôlés. |
| Résultat exploratoire | L'expérience a été exécutée, mais un choix a été fait directement sur la validation ou sans protocole OOF complet. |
| Proposition | Idée non encore mesurée ; elle ne doit pas être présentée comme un résultat. |

Le split `test` n'a pas été utilisé pour sélectionner les modèles avancés ni pour
produire les scores finaux rapportés ici.

## 3. Chronologie générale

### 3.1. 5 août 2026 — publication du dépôt étudiant

Les premiers commits ont publié le code du projet et mis à jour les liens vers les
ressources. L'objectif annoncé dans [README.md](README.md) est de passer de l'ADN
brut à un petit classifieur distillé depuis un modèle de fondation génomique.

### 3.2. 6 au 8 août 2026 — ajout des données et du matériel complémentaire

Le dépôt a été enrichi avec :

- les scripts de construction du dataset ;
- les instructions de téléchargement des embeddings ;
- les notebooks des cinq jours ;
- les ressources pédagogiques et le bonus d'interprétabilité.

### 3.3. 10 août 2026 — complétion locale du parcours pédagogique

Le commit local le plus récent a complété plusieurs TODO du parcours : chargement
des données, représentations k-mers et one-hot, modèles, évaluations et outils de
visualisation. Les expériences avancées postérieures au 10 août sont encore des
modifications locales non regroupées dans un nouveau commit.

### 3.4. 12 août 2026 — expérimentations avancées du Jour 3

Après le constat que la première distillation plafonnait autour de 84 %, plusieurs
expériences ont été lancées :

1. feature engineering codon/phase et modèles tabulaires ;
2. CNN 1D multi-échelle ;
3. distillation hybride sur toutes les données, avec logits Evo2 disponibles sur
   seulement 4 000 lignes ;
4. vote de plusieurs modèles ;
5. retour à la contrainte stricte des 4 000 embeddings ;
6. mise en place de la distillation OOF groupée par organisme.

Les sections suivantes reprennent tout ce parcours dans l'ordre logique.

---

## 4. Étape 0 — contexte biologique et construction des données

### Objectif

Une région codante contient de l'information traduisible en protéine. Une région
non codante n'est pas utilisée de cette manière. Le problème est formulé comme une
classification binaire de fenêtres d'ADN.

### Sources biologiques

Le notebook [day1/00_biology_intro_and_data_setup.ipynb](day1/00_biology_intro_and_data_setup.ipynb)
introduit :

- **FASTA** : séquence complète du génome ;
- **GFF** : annotations indiquant notamment les coordonnées des CDS ;
- **CDS** : région codante ;
- **fenêtre positive** : fenêtre extraite d'une CDS ;
- **fenêtre négative** : fenêtre extraite hors des CDS.

Le script [1-Introduction/prepare_data.py](1-Introduction/prepare_data.py) et les
scripts de `2-data/` construisent les CSV traités.

### Données disponibles

| Split | Nombre de lignes brut | Usage |
|---|---:|---|
| Train | 58 552 | Apprentissage et expériences internes |
| Validation | 10 374 | Comparaison des modèles sur tous les organismes de validation |
| Test | 18 448 | Évaluation finale, volontairement non consultée dans les expériences avancées |

Les classes sont approximativement équilibrées. Les organismes du train et de la
validation sont disjoints : 20 organismes dans le train et 4 dans la validation.

### Sous-échantillon pédagogique

La fonction [day1/src/data.py](day1/src/data.py) utilise par défaut
`max_rows = 4000`. Elle prélève un échantillon équilibré et déterministe avec la
seed 42. Cette contrainte permet de comparer les baselines aux 4 000 embeddings
Evo2 disponibles.

Une distinction importante apparaît donc dès le début :

- parcours pédagogique : 4 000 train et généralement 1 000 validations alignées
  avec les embeddings ;
- expériences supervisées avancées : jusqu'à 58 550 train et 10 373 validations
  après contrôle de longueur.

---

## 5. Jour 1 — représentations simples et modèles de référence

Le notebook principal est
[day1/01_kmer_and_cnn_baselines.ipynb](day1/01_kmer_and_cnn_baselines.ipynb).

### 5.1. Baseline k-mers

Une séquence est transformée en fréquences de mots d'ADN de longueur `k`.
L'expérience utilise les 4-mers, donc `4^4 = 256` caractéristiques.

Deux modèles classiques ont été testés :

| Modèle | Accuracy | F1 | Paramètres représentés | Latence enregistrée |
|---|---:|---:|---:|---:|
| Régression logistique sur 4-mers | 78,85 % | 80,80 % | 257 | 0,0725 ms |
| Arbre de décision | 71,80 % | 72,87 % | 683 | 0,0568 ms |

La régression logistique apprend une frontière linéaire dans l'espace des
fréquences. L'arbre de décision apprend des règles de seuil, mais généralise moins
bien dans cette expérience.

### 5.2. Baseline CNN one-hot

Chaque nucléotide est encodé sur quatre canaux `A/C/G/T`. Le CNN reçoit donc un
tenseur de forme `(4, 200)` par séquence. L'architecture contient deux convolutions
1D de noyau 9 et une tête linéaire.

| Modèle | Accuracy | F1 | Paramètres | Latence enregistrée |
|---|---:|---:|---:|---:|
| CNN one-hot | 83,33 % | 84,13 % | 10 465 | 0,1812 ms |

Le CNN dépasse les modèles k-mers simples parce qu'il conserve l'ordre local des
nucléotides et apprend lui-même des motifs utiles.

### Conclusion du Jour 1

Le premier niveau de performance se situe entre 72 % et 83 %. Le CNN est la
meilleure baseline du jour, mais il doit apprendre ses représentations à partir de
seulement quelques milliers d'exemples.

---

## 6. Jour 2 — embeddings Evo2 et teacher MLP

Le notebook est
[day2/02_evo2_embeddings_and_classifier.ipynb](day2/02_evo2_embeddings_and_classifier.ipynb).

### Principe

Evo2 transforme une séquence en embedding de dimension 4 096. Ces embeddings sont
pré-calculés et stockés dans :

- `2-data/embeddings/train.npz` : 4 000 exemples ;
- `2-data/embeddings/val.npz` : 1 000 exemples ;
- `2-data/embeddings/test.npz` : réservé au test.

Evo2 reste gelé. Seule une petite tête `MLPHead` est entraînée :

```text
Embedding Evo2 4096
        -> Linear(4096, 128)
        -> ReLU
        -> Dropout(0,1)
        -> Linear(128, 1)
        -> logit binaire
```

La tête possède 524 545 paramètres entraînables. Son checkpoint est sauvegardé
dans `2-data/models/teacher_mlp.pt` et occupe environ 2,1 Mo.

### Résultat exécuté

| Modèle | Accuracy | F1 |
|---|---:|---:|
| Evo2 gelé + tête MLP | 95,80 % | 95,85 % |

### Interprétation

Le saut de 83,33 % à 95,80 % montre que l'embedding Evo2 contient une
représentation très informative pour la tâche. La tête MLP ne crée pas seule cette
connaissance ; elle apprend surtout la frontière de décision dans l'espace Evo2.

Ce teacher est précis, mais il suppose que les embeddings Evo2 ont déjà été
calculés. L'objectif du Jour 3 est donc d'obtenir un modèle autonome et plus léger.

---

## 7. Jour 3 initial — première distillation

Le notebook d'origine est
[day3/03_knowledge_distillation.ipynb](day3/03_knowledge_distillation.ipynb).

### Student d'origine

Le student reçoit les 256 fréquences de 4-mers et utilise un MLP :

```text
256 fréquences de 4-mers -> couche cachée -> sortie binaire
```

La configuration retenue dans le benchmark possède 16 513 paramètres.

### Perte de distillation

La perte combine les labels réels et les probabilités du teacher :

```text
L = alpha * L_hard + (1 - alpha) * T² * L_soft
```

- `L_hard` : erreur par rapport au vrai label ;
- `L_soft` : erreur par rapport à la probabilité du teacher ;
- `alpha` : poids du label réel ;
- `T` : température contrôlant l'adoucissement des probabilités.

### Résultat initial

Une première exécution a donné environ :

| Modèle | Accuracy | F1 |
|---|---:|---:|
| Student distillé initial | 84,30 % | 84,47 % |

Une recherche de 180 combinaisons a ensuite retenu dans la sortie actuelle :

```text
d_hidden = 64
alpha = 0,25
T = 2
Accuracy = 84,10 %
F1 = 84,46 %
```

Le benchmark homogène du Jour 4 utilise 84,00 % d'accuracy et 84,31 % de F1 pour
ce student.

### Faiblesses identifiées

1. Les logits teacher étaient produits sur les mêmes exemples que ceux ayant servi
   à entraîner le teacher. Ils pouvaient donc refléter de la mémorisation.
2. La validation officielle était utilisée pour comparer de nombreuses
   configurations. Le score final était donc optimisé sur cette validation.
3. Les seeds n'étaient pas toutes fixées, ce qui rendait le meilleur réglage
   variable entre les exécutions.
4. La grille contenait `alpha = 10`, qui sort du domaine normal `[0,1]`. Le terme
   `(1 - alpha)` devenait négatif, provoquant des pertes négatives et des modèles à
   50 % d'accuracy.
5. Le student n'avait que les 4-mers. La représentation était probablement le
   principal goulot d'étranglement.

Cette expérience a été utile : elle a montré qu'un teacher fort ne garantit pas
une bonne distillation si le protocole et la représentation du student sont trop
faibles.

---

## 8. Première tentative d'amélioration — grand feature engineering et ensemble

Le notebook
[day3/03_test_distillation_experiments.ipynb](day3/03_test_distillation_experiments.ipynb)
a essayé une représentation beaucoup plus large :

- k-mers de longueurs 3, 4, 5 et 6 ;
- taux GC ;
- 16 dinucléotides ;
- entropie de la séquence ;
- longueur d'ORF simplifiée ;
- sélection des 2 000 variables les plus importantes ;
- Random Forest, ExtraTrees, régression logistique, Gradient Boosting et MLP ;
- vote moyen des cinq modèles.

### Résultat enregistré dans l'état actuel

Le code charge actuellement `max_rows=4000`, malgré un commentaire indiquant
« totalité des 58 552 ». La sortie enregistrée utilise 4 000 train et 4 000
validations :

| Modèle | Accuracy | F1 |
|---|---:|---:|
| Random Forest | 85,85 % | 86,45 % |
| ExtraTrees | 85,93 % | 86,58 % |
| Régression logistique | 82,30 % | 82,78 % |
| Gradient Boosting | 86,50 % | 86,86 % |
| Deep MLP | 85,30 % | 85,55 % |
| Vote uniforme | 86,60 % | 86,99 % |

### Ce que cette tentative a apporté

- elle a confirmé que plusieurs familles de modèles capturaient des signaux
  complémentaires ;
- elle a encouragé un feature engineering plus biologique ;
- elle a montré qu'ajouter des milliers de k-mers n'était pas suffisant pour
  atteindre proprement 90 % sous la contrainte de 4 000 exemples.

### Limite actuelle du notebook

Son titre annonce plus de 90,7 %, alors que sa sortie actuelle affiche 86,6 %.
Le commentaire « totalité des données » contredit `max_rows=4000`. Ce notebook ne
doit donc pas être utilisé comme preuve finale sans nettoyage et réexécution.

---

## 9. Expériences avancées menées en parallèle

### 9.1. Axe A — caractéristiques codon/phase

L'hypothèse était que la différence codant/non codant dépend moins d'un très grand
vocabulaire de motifs que de la structure biologique des cadres de lecture.

Le fichier [day3/src/advanced_features.py](day3/src/advanced_features.py) construit
un vecteur de 761 dimensions :

| Bloc | Dimensions | Rôle |
|---|---:|---|
| k-mers globaux de longueur 1 à 4 | 340 | Composition et motifs locaux |
| Codons dans 3 phases, brin direct | 192 | Structure du cadre de lecture direct |
| Codons dans 3 phases, reverse-complement | 192 | Structure du brin opposé |
| Résumés start/stop/ORF | 24 | Starts, stops, longueur sans stop, premier stop |
| Fréquences A/C/G/T selon la phase | 12 | Périodicité de trois nucléotides |
| Écart GC entre phases | 1 | Intensité résumée de la périodicité GC |
| **Total** | **761** | |

Le taux GC de phase ne correspond pas au 2-mer `GC`. Le 2-mer mesure combien de
fois `G` est immédiatement suivi de `C`. Le contenu GC mesure `freq(G)+freq(C)`.

#### Contrôle de qualité

Trois séquences n'avaient pas la longueur attendue de 200 :

- 2 lignes exclues du train ;
- 1 ligne exclue de la validation.

Les expériences utilisent donc 58 550 train et 10 373 validations.

#### Modèles et résultats sur la validation complète

| Modèle | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Régression logistique enrichie | 92,26 % | 92,31 % | 97,44 % |
| HistGradientBoosting sur 37 résumés | 95,34 % | 95,42 % | 98,48 % |
| ExtraTrees, seed 42 | 95,85 % | 95,94 % | 98,80 % |

La stabilité d'ExtraTrees sur les seeds 7, 42 et 123 était :

```text
Accuracy moyenne = 95,848 %
Écart-type = 0,032 point
```

L'utilisation du reverse-complement apporte environ 0,20 point d'accuracy par
rapport à la version directe seule.

#### Audit des chevauchements

Les organismes train et validation sont disjoints, et aucune coordonnée exacte ne
se chevauche. Cependant :

- 848 séquences uniques exactes existent dans les deux splits ;
- 863 séquences uniques deviennent communes après canonicalisation par
  reverse-complement ;
- 872 lignes de validation sont concernées ;
- ExtraTrees obtient 100 % sur ce sous-ensemble, signe clair de mémorisation
  possible.

Le score prudent est donc calculé sur les 9 501 fenêtres canoniques réellement
inédites :

| Modèle | Surface | Accuracy | F1 | AUC |
|---|---|---:|---:|---:|
| ExtraTrees seed 42 | Validation inédite | **95,47 %** | **95,86 %** | **98,57 %** |

Ce résultat reste le meilleur score individuel supervisé audité parmi les
variantes mesurées sur toutes les données.

#### Vote exploratoire

Un vote pondéré `0,60 ExtraTrees + 0,20 HGB + 0,20 LR` a atteint 95,82 % sur la
validation inédite. Mais ces poids ont été examinés directement sur la validation
et n'ont pas été appris en OOF. Ce score reste donc exploratoire.

Le notebook associé est
[day3/03_codon_phase_advanced_experiments.ipynb](day3/03_codon_phase_advanced_experiments.ipynb).

### 9.2. Axe B — CNN 1D multi-échelle

Un CNN séparé a été entraîné directement sur les séquences one-hot de longueur 200.
Son architecture combine :

- convolutions parallèles de noyaux 3, 9 et 15 ;
- bloc résiduel de noyau 5 ;
- convolution dilatée de noyau 7 ;
- average pooling et max pooling ;
- tête de classification.

| Mesure | Valeur |
|---|---:|
| Paramètres | 116 169 |
| Meilleure époque | 13 |
| Époques exécutées | 19, avec early stopping |
| Accuracy | 93,12 % |
| F1 | 93,22 % |
| Temps total local | 202,3 s |

Le modèle commence à surapprendre après l'époque 13. Cette approche dépasse le CNN
simple du Jour 1, mais reste derrière ExtraTrees codon/phase.

Le script `/tmp/eeia_cnn_sequence.py` est un artefact expérimental temporaire, pas
un composant durable du pipeline OOF.

### 9.3. Axe C — distillation hybride sur toutes les données

Cette expérience utilisait :

- les labels durs sur les 58 550 séquences ;
- les logits teacher seulement sur les 4 000 exemples alignés, soit 6,83 % du
  train ;
- 461 caractéristiques enrichies ;
- plusieurs poids de distillation.

| Configuration | Accuracy | F1 | Écart vs hard |
|---|---:|---:|---:|
| Hard, sans KD | 95,09 % | 95,10 % | référence |
| KD alpha 0,98, T=2 | 95,04 % | 95,04 % | -0,06 point |
| KD alpha 0,98, T=4 | 94,97 % | 94,97 % | -0,13 point |
| KD alpha 0,50, T=4 | 94,15 % | 94,19 % | -0,94 point |

Aucune variante KD n'a battu le témoin. Le teacher atteignait 100 % sur les 4 000
exemples qu'il avait vus, ce qui indiquait surtout des cibles surconfiantes ou
mémorisées. Cette expérience a motivé le passage aux logits OOF.

### 9.4. Ce que les expériences parallèles ont enseigné

1. La représentation codon/phase apporte plus que la complexification du CNN dans
   cette tâche tabulaire.
2. La distillation n'aide pas automatiquement : la qualité et l'indépendance des
   cibles teacher sont essentielles.
3. Utiliser tout le train facilite fortement l'objectif de 90 %, mais ne répond
   plus au défi volontaire des 4 000 embeddings.
4. Un audit de doublons exacts et reverse-complements est indispensable avant de
   qualifier une validation de « propre ».

---

## 10. Retour à la contrainte des 4 000 embeddings

L'objectif a ensuite été reformulé : obtenir environ 90 % sans utiliser les 58 552
séquences, uniquement avec :

- les 4 000 séquences possédant un embedding Evo2 pour l'entraînement ;
- les 1 000 séquences possédant un embedding Evo2 pour la validation ;
- un petit modèle utilisable sans Evo2 à l'inférence.

### Vérifications intermédiaires sur 4 000 exemples

Avec les 761 caractéristiques et des modèles classiques, des essais ont montré :

| Modèle | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Régression logistique | 87,10 % | 87,24 % | 94,34 % |
| SVC RBF | 89,40 % | 89,48 % | 96,02 % |
| HGB sur 37 résumés | 93,00 % | 93,18 % | 97,48 % |
| ExtraTrees | 93,40 % | 93,68 % | 98,12 % |

Cela a confirmé qu'il était possible de dépasser 90 % avec 4 000 exemples, mais
l'objectif pédagogique restait de construire un **petit MLP distillé**, pas de
remplacer le student par une forêt d'arbres.

---

## 11. Approche finale — distillation OOF groupée par organisme

Le notebook de référence est
[day3/03_serious_oof_distillation.ipynb](day3/03_serious_oof_distillation.ipynb).

### 11.1. Question expérimentale

Peut-on transférer l'information contenue dans les embeddings Evo2 vers un petit
MLP autonome, avec seulement 4 000 exemples, sans donner au student des sorties
teacher obtenues par mémorisation ?

### 11.2. Entrées des deux modèles

```text
Pendant l'entraînement

Séquence ADN
  |-- embedding Evo2 4096 -> teacher MLP -> logit OOF ---------|
  |                                                           |
  `-- 761 caractéristiques codon/phase -> student MLP <-------|
                                              ^
                                              |
                                         vrai label

À l'inférence

Séquence ADN -> 761 caractéristiques -> student MLP -> prédiction
```

Evo2 et le teacher disparaissent donc entièrement du chemin d'inférence.

### 11.3. Architecture du student

```text
761 entrées
  -> Linear(761, 64)
  -> LayerNorm(64)
  -> GELU
  -> Dropout(0,2)
  -> Linear(64, 1)
```

Calcul du nombre de paramètres :

```text
761 * 64 + 64 = 48 768
LayerNorm        =    128
64 * 1 + 1       =     65
Total            = 48 961 paramètres
```

### 11.4. Pourquoi l'out-of-fold est nécessaire

Une cible teacher est valide seulement si le teacher qui la produit n'a pas
appris sur cet exemple. Nous utilisons donc `GroupKFold` avec l'organisme comme
groupe :

1. quatre organismes sont gardés de côté ;
2. un teacher neuf est entraîné sur les seize autres ;
3. il produit les logits des organismes gardés de côté ;
4. l'opération est répétée cinq fois ;
5. chaque séquence reçoit exactement un logit hors échantillon.

La séparation par organisme évite qu'un teacher profite des signatures très
proches d'un même organisme des deux côtés du split.

### 11.5. Sélection interne sans ouvrir la validation finale

Les 4 000 exemples train sont séparés en :

| Sous-ensemble | Exemples | Organismes | Usage |
|---|---:|---:|---|
| Développement | 3 246 | 16 | Entraîner les candidates et produire les cibles OOF |
| Réglage interne | 754 | 4 | Choisir alpha, T, époque et seuil |

La validation officielle de 1 000 exemples reste fermée pendant cette sélection.

Configuration choisie :

```text
alpha = 0,5
température T = 1
57 époques
seuil = 0,265
```

`T = 1` reste une distillation : les probabilités du teacher sont utilisées comme
cibles douces, mais aucun adoucissement supplémentaire n'a été utile.

### 11.6. Qualité des cibles teacher OOF

Les cinq teachers temporaires atteignent globalement :

| Mesure | Valeur |
|---|---:|
| Accuracy OOF teacher | 97,75 % |
| F1 OOF teacher | 97,76 % |
| Exemples avec exactement un logit OOF | 4 000 / 4 000 |

Ce chiffre mesure la qualité des cibles sur le train cross-fitté ; ce n'est pas le
score final du student.

### 11.7. Évaluation finale sur trois seeds

Les hyperparamètres et les seuils sont fixés avant l'ouverture de la validation.
Le même student est ensuite entraîné avec et sans KD pour les seeds 7, 42 et 123.

| Seed | Hard accuracy | KD accuracy | Hard F1 | KD F1 | Hard AUC | KD AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 89,20 % | 90,20 % | 89,62 % | 90,56 % | 95,35 % | 96,43 % |
| 42 | 89,80 % | 90,30 % | 90,19 % | 90,68 % | 95,30 % | 96,21 % |
| 123 | 90,40 % | 89,90 % | 90,73 % | 90,32 % | 95,25 % | 96,24 % |

Synthèse :

| Modèle | Accuracy moyenne | F1 moyen | AUC moyenne | Écart-type accuracy |
|---|---:|---:|---:|---:|
| Même MLP sans distillation | 89,80 % | 90,18 % | 95,30 % | 0,49 point |
| MLP avec distillation OOF | **90,13 %** | **90,52 %** | **96,29 %** | **0,17 point** |

La KD apporte :

- `+0,33` point d'accuracy moyenne ;
- `+0,34` point de F1 moyen ;
- `+0,99` point d'AUC moyenne ;
- une variance plus faible entre seeds.

Deux seeds KD sur trois dépassent 90 %. La troisième atteint 89,9 %. La formulation
rigoureuse est donc : **90,13 % en moyenne**, pas « toujours plus de 90 % ».

Un audit auxiliaire du sous-ensemble 4 000/1 000 a identifié 993 validations
canoniques inédites, donc 7 fenêtres ayant un équivalent exact ou
reverse-complement dans le train. Le notebook OOF rapporte actuellement ses scores
sur les 1 000 lignes et ne donne pas le score séparé du MLP final sur ces 993
fenêtres. L'effet potentiel est limité en volume, mais cette distinction doit être
conservée dans une présentation rigoureuse.

### 11.8. Ce que le résultat prouve et ne prouve pas

Il prouve que, pour le même student, les mêmes caractéristiques et les mêmes 4 000
séquences, les cibles OOF améliorent en moyenne la décision et surtout l'AUC.

Il ne prouve pas :

- que tout Evo2 a été compressé dans le student ;
- que le gain de 0,33 point est statistiquement significatif avec seulement trois
  seeds ;
- que l'accuracy dépassera 90 % à chaque réentraînement ;
- que le feature engineering n'a aucun rôle : la comparaison hard/KD isole le
  gain de KD, mais le niveau global de 90 % dépend fortement des 761 variables.

---

## 12. Jour 4 — compromis précision, taille et latence

Le notebook
[day4/04_compression_analysis_and_wrapup.ipynb](day4/04_compression_analysis_and_wrapup.ipynb)
compare le student original et l'approche OOF.

| Modèle | Accuracy | F1 | Paramètres | Taille | Features | Forward CPU | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jour 3 original, 4-mers + MLP | 84,00 % | 84,31 % | 16 513 | 0,0684 Mo | 0,0399 ms | 0,0124 ms | 0,0523 ms |
| Codon/phase + KD OOF | 90,13 % | 90,52 % | 48 961 | 0,1986 Mo | 0,3879 ms | 0,0190 ms | 0,4069 ms |

Le nouveau student apporte 6,13 points d'accuracy par rapport au student original,
mais utilise environ 2,96 fois plus de paramètres et 7,78 fois plus de temps total.

Environ 95,3 % de la latence du nouveau pipeline vient de l'extraction des 761
caractéristiques, et non du MLP. La meilleure optimisation future viserait donc
le feature engineering, pas le forward du réseau.

Ces latences sont des médianes mesurées localement sur CPU et dépendent du matériel.

---

## 13. Jour 5 — bonus d'interprétabilité

Le notebook
[day5/bonus_sparse_autoencoder_interpretability.ipynb](day5/bonus_sparse_autoencoder_interpretability.ipynb)
propose un autoencodeur parcimonieux pour chercher des dimensions interprétables
dans les activations d'un modèle génomique.

Le parcours prévu est :

1. charger des activations Evo2 ;
2. entraîner un autoencodeur Top-K ;
3. extraire ses features le long d'un génome sonde ;
4. classer les features selon leur séparation codant/non codant ;
5. rechercher une périodicité de trois nucléotides.

Dans l'état actuel du dépôt, les neuf cellules de code de ce notebook ne sont pas
exécutées. Cette piste reste donc un bonus non validé et ne participe pas aux
résultats finaux.

---

## 14. Carte des fichiers importants

| Fichier | Rôle actuel |
|---|---|
| [README.md](README.md) | Présentation du parcours et instructions d'environnement |
| [1-Introduction/guide.md](1-Introduction/guide.md) | Contexte biologique et organisation de la semaine |
| [day1/00_biology_intro_and_data_setup.ipynb](day1/00_biology_intro_and_data_setup.ipynb) | FASTA, GFF et construction du dataset |
| [day1/01_kmer_and_cnn_baselines.ipynb](day1/01_kmer_and_cnn_baselines.ipynb) | Baselines 4-mers, arbre et CNN one-hot |
| [day2/02_evo2_embeddings_and_classifier.ipynb](day2/02_evo2_embeddings_and_classifier.ipynb) | Teacher MLP sur embeddings Evo2 |
| [day3/03_knowledge_distillation.ipynb](day3/03_knowledge_distillation.ipynb) | Distillation pédagogique d'origine et grille exploratoire |
| [day3/03_test_distillation_experiments.ipynb](day3/03_test_distillation_experiments.ipynb) | Première tentative de grand feature engineering ; état incohérent à nettoyer |
| [day3/src/advanced_features.py](day3/src/advanced_features.py) | Construction exacte des 761 caractéristiques |
| [day3/03_codon_phase_advanced_experiments.ipynb](day3/03_codon_phase_advanced_experiments.ipynb) | Expérience supervisée complète et audit des overlaps |
| [day3/03_serious_oof_distillation.ipynb](day3/03_serious_oof_distillation.ipynb) | Approche finale limitée aux 4 000 embeddings |
| [day3/src/benchmark.py](day3/src/benchmark.py) | Paramètres, taille, feature latency et forward latency |
| [day4/04_compression_analysis_and_wrapup.ipynb](day4/04_compression_analysis_and_wrapup.ipynb) | Comparaison efficacité du student original et du student OOF |
| [day4/src/viz.py](day4/src/viz.py) | Graphique accuracy/latence/taille |
| [day5/bonus_sparse_autoencoder_interpretability.ipynb](day5/bonus_sparse_autoencoder_interpretability.ipynb) | Bonus SAE non exécuté |

---

## 15. État technique actuel du dépôt

### Notebooks propres et exécutés sans cellule en erreur enregistrée

- `day2/02_evo2_embeddings_and_classifier.ipynb` ;
- `day3/03_knowledge_distillation.ipynb` ;
- `day3/03_serious_oof_distillation.ipynb` ;
- `day4/04_compression_analysis_and_wrapup.ipynb`.

### Notebooks partiellement exécutés

- `day1/00_biology_intro_and_data_setup.ipynb` : 3 cellules de code exécutées sur 5 ;
- `day1/01_kmer_and_cnn_baselines.ipynb` : 7 cellules exécutées sur 8 ;
- `day5/bonus_sparse_autoencoder_interpretability.ipynb` : aucune cellule exécutée.

### Notebooks avec un marqueur de kernel crash enregistré

- `day3/03_codon_phase_advanced_experiments.ipynb` ;
- `day3/03_test_distillation_experiments.ipynb`.

Les résultats codon/phase ont bien été exécutés et conservés, mais le notebook a
ensuite reçu des cellules de démonstration et n'est plus dans un état proprement
réexécuté de haut en bas. Avant livraison finale, il faudra nettoyer les doublons,
redémarrer le kernel et exécuter toutes les cellules dans l'ordre.

### État Git

Le dépôt est sur la branche `main`. Les expériences récentes sont un mélange de
fichiers indexés, modifiés après indexation et non suivis. Aucun commit ne rassemble
encore proprement l'approche codon/phase, la KD OOF, le benchmark et le bilan du
Jour 4.

---

## 16. Résultats à retenir

### Parcours pédagogique sur le petit échantillon

| Étape | Modèle | Accuracy |
|---|---|---:|
| Jour 1 | Régression logistique 4-mers | 78,85 % |
| Jour 1 | CNN one-hot | 83,33 % |
| Jour 2 | Evo2 + teacher MLP | 95,80 % |
| Jour 3 initial | Student 4-mers distillé | environ 84 % |
| Jour 3 final | Student codon/phase + KD OOF | **90,13 % moyenne** |

### Expériences sur toutes les données

| Modèle | Accuracy rapportable | Statut |
|---|---:|---|
| CNN 1D multi-échelle | 93,12 % | Mesuré, validation complète |
| MLP enrichi sans KD | 95,09 % | Mesuré, validation complète |
| ExtraTrees codon/phase | 95,47 % | Mesuré sur validation canonique inédite |
| Vote ET/HGB/LR | 95,82 % | Exploratoire, poids non OOF |

---

## 17. Formulations sûres pour présenter le projet

### Ce que nous pouvons affirmer

> Nous avons construit un student MLP de 48 961 paramètres qui utilise 761
> caractéristiques biologiques calculées directement depuis l'ADN. Avec seulement
> 4 000 exemples et une distillation OOF groupée par organisme, il atteint 90,13 %
> d'accuracy moyenne et n'a plus besoin d'Evo2 à l'inférence.

> La comparaison contrôlée avec le même student sans KD montre un gain moyen de
> 0,33 point d'accuracy, 0,99 point d'AUC et une réduction de la variance entre les
> seeds.

> Sur toutes les données supervisées, ExtraTrees avec les caractéristiques
> codon/phase atteint 95,47 % sur les fenêtres de validation sans équivalent exact
> ou reverse-complement dans le train.

### Ce qu'il ne faut pas affirmer

- « Le student a compressé tout Evo2 » ;
- « La distillation seule nous a fait passer de 84 % à 90 % » ;
- « Le modèle dépasse toujours 90 % » ;
- « Le vote à 95,82 % est notre score final validé » ;
- « Le test confirme ces résultats » ;
- « Tous les notebooks se réexécutent actuellement sans erreur ».

### Formulation complète et rigoureuse

> L'amélioration de 84 % à 90,13 % vient principalement d'une meilleure
> représentation biologique du student, complétée par une distillation OOF. Pour
> isoler l'effet de la distillation, nous avons comparé exactement le même MLP avec
> et sans cibles teacher : le gain moyen propre à la KD est de 0,33 point
> d'accuracy et de 0,99 point d'AUC.

---

## 18. Limites et prochaines actions recommandées

1. Réexécuter proprement les deux notebooks avancés contenant un marqueur de crash.
2. Corriger le titre, les commentaires et le sampling incohérents de
   `03_test_distillation_experiments.ipynb`.
3. Ajouter des intervalles de confiance ou davantage de splits groupés pour juger
   la significativité du gain KD de 0,33 point.
4. Remplacer le seul holdout interne de 754 exemples par une validation imbriquée
   répétée si le temps de calcul le permet.
5. Ajouter au notebook OOF le score du MLP final sur les 993 validations
   canoniques inédites, en complément du score principal sur 1 000 lignes.
6. Optimiser `advanced_feature_matrix`, car l'extraction représente 95,3 % de la
   latence du student final.
7. Conserver le split test fermé jusqu'au choix définitif d'un seul pipeline.
8. Regrouper les modifications validées dans un commit propre après une exécution
   de bout en bout et une revue des notebooks.

## Conclusion

Le projet a progressé d'une baseline CNN à 83,33 % vers un teacher Evo2 à 95,80 %,
puis vers un student autonome à 90,13 % avec seulement 4 000 exemples. Le principal
apport technique n'est pas une simple augmentation de la taille du réseau : c'est
la combinaison de caractéristiques biologiques structurées, d'un protocole OOF
groupé par organisme et d'une comparaison honnête contre le même student sans KD.

En parallèle, l'expérience sur toutes les données a montré que les caractéristiques
codon/phase sont très fortes pour cette tâche, avec 95,47 % sur la validation
canonique inédite. L'audit des chevauchements a toutefois rappelé qu'un score élevé
ne vaut que si la surface d'évaluation est clairement définie et contrôlée.
