# AGENTS.md — guide de relève pour les agents de code

Ce fichier s’applique à **tout le dépôt**. Un éventuel `AGENTS.md` placé dans un
sous-dossier peut préciser des règles locales, mais ne doit pas affaiblir les
invariants scientifiques, d’attribution ou de sécurité définis ici.

## 1. Mission du dépôt

Ce dépôt provient du projet collectif de la semaine 4 de l’**EEIA 2026**. Il étudie
une tâche de classification binaire de fenêtres d’ADN bactérien de 200 nucléotides :

- `1` : fenêtre provenant d’une région codante (`CDS`) ;
- `0` : fenêtre provenant d’une région non codante/intergénique.

Le projet compare des baselines k-mers/CNN, un teacher fondé sur des embeddings
Evo2, puis un student compact entraîné avec une distillation out-of-fold groupée
par organisme.

Le problème **n’est pas** la classification des fonctions de protéines. Ne pas
réintroduire cette formulation imprécise dans le README, la documentation, les
issues, les figures ou les publications.

## 2. À lire avant toute modification

Lire, dans cet ordre :

1. [`README.md`](README.md) — vue générale et état publiable ;
2. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — protocole expérimental ;
3. [`docs/RESULTS.md`](docs/RESULTS.md) — résultats et niveaux de preuve ;
4. [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — menaces à la validité ;
5. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — contrat de reproduction ;
6. [`HISTORIQUE_DETAILLE_DU_PROJET.md`](HISTORIQUE_DETAILLE_DU_PROJET.md) — historique complet ;
7. [`CONTRIBUTORS.md`](CONTRIBUTORS.md) — attribution et contributions à confirmer.

La roadmap de release est suivie dans l’issue GitHub `#1`.

## 3. Sources de vérité

### Code public et réutilisable

La logique appelée par les futurs outils doit vivre dans :

```text
src/bioai/
```

Les notebooks peuvent expliquer et orchestrer des expériences, mais ne doivent
pas devenir la seule implémentation d’une logique critique.

### Contrats stables actuels

- version des caractéristiques : `codon-phase-761-v1` ;
- dimension du student : `761` entrées ;
- architecture de référence : `761 -> 64 -> LayerNorm -> GELU -> Dropout -> 1` ;
- nombre de paramètres attendu : `48 961` ;
- format de checkpoint : état du modèle + architecture + version des features +
  seuil + métadonnées ;
- séparation scientifique principale : groupes définis par `organism` ;
- séquences équivalentes pour l’audit : séquence exacte ou reverse-complement.

Toute modification de l’ordre, du sens ou du nombre des caractéristiques doit :

1. créer une nouvelle version de features ;
2. conserver l’ancienne version pour charger les anciens checkpoints ;
3. ajouter des tests de compatibilité/migration ;
4. mettre à jour la model card, les configs et la documentation.

Ne jamais modifier silencieusement `codon-phase-761-v1`.

## 4. Invariants scientifiques non négociables

### 4.1. Le split test reste scellé

Ne pas charger, inspecter, optimiser ou rapporter de métriques sur le split `test`
sans une demande explicite du responsable du projet confirmant que :

- un pipeline unique a été gelé ;
- tous les hyperparamètres et seuils sont fixés ;
- l’équipe accepte l’ouverture unique du test ;
- le protocole et le commit sont archivés avant l’exécution.

Les tests logiciels doivent utiliser de petites données synthétiques, jamais le
split test biologique.

### 4.2. Éviter les fuites de données

- grouper les splits par organisme lorsque le protocole l’exige ;
- auditer les doublons exacts et reverse-complements ;
- produire les cibles teacher de distillation hors échantillon ;
- ne pas choisir un seuil, une seed ou un hyperparamètre sur la surface ensuite
  présentée comme évaluation finale ;
- distinguer clairement validation complète et validation canoniquement inédite.

### 4.3. Ne pas embellir les résultats

Les formulations actuellement acceptables sont :

- teacher Evo2 + tête MLP : `95,80 %` d’accuracy sur la surface documentée ;
- student codon/phase + KD OOF : `90,13 %` d’accuracy **moyenne sur trois seeds** ;
- même student sans KD : `89,80 %` en moyenne ;
- gain KD observé : `+0,33` point d’accuracy moyenne et `+0,99` point de ROC-AUC ;
- ExtraTrees codon/phase : `95,47 %` sur 9 501 fenêtres canoniques inédites.

Ne pas transformer ces faits en :

- « score final de test » ;
- « le modèle dépasse toujours 90 % » ;
- « la distillation seule fait passer de 84 % à 90 % » ;
- « tout Evo2 a été compressé » ;
- « le gain KD est statistiquement prouvé ».

Une nouvelle métrique ne peut être ajoutée à la documentation que si son script,
son commit, sa surface d’évaluation, ses seeds et son artefact source sont traçables.

### 4.4. Conserver les surfaces comparables

Ne pas comparer directement comme s’ils avaient vu les mêmes données :

- le parcours limité aux 4 000 exemples alignés avec les embeddings Evo2 ;
- les expériences supervisées utilisant jusqu’à 58 550 exemples ;
- les résultats sur toutes les validations ;
- les résultats après exclusion des équivalents canoniques.

Chaque tableau doit indiquer la surface et le protocole.

## 5. Données, artefacts et secrets

### Données volumineuses

Ne pas committer :

- embeddings Evo2 `.npz` volumineux ;
- checkpoints ou artefacts dépassant les limites raisonnables de Git ;
- caches, environnements virtuels, sorties temporaires ;
- fichiers contenant des clés NVIDIA ou autres secrets.

Respecter `.gitignore`. Pour les gros artefacts, utiliser la stratégie décidée par
le projet : Hugging Face, Zenodo, DVC, Git LFS ou stockage externe documenté.

### Provenance

Avant toute release, documenter pour chaque ressource externe :

- source et accession ;
- version/date ;
- licence ou conditions d’utilisation ;
- checksum SHA256 ;
- script ayant produit l’artefact dérivé.

Ne pas inventer une provenance manquante.

### Secrets

- utiliser des variables d’environnement ;
- ne jamais écrire de clé dans un notebook, un test ou un exemple ;
- ne jamais afficher une clé dans les logs ;
- en cas de secret détecté, arrêter la tâche, le signaler et recommander sa
  révocation avant de poursuivre.

## 6. Attribution et propriété intellectuelle

Le matériel pédagogique initial vient de **Généreux Akotenou** et comporte des
en-têtes MIT à conserver. Le projet a été réalisé par une équipe de participants
avec des superviseurs listés dans `CONTRIBUTORS.md`.

Règles :

- ne pas supprimer les en-têtes d’attribution ;
- ne pas présenter le dépôt comme le travail exclusif d’une personne ;
- ne pas remplir arbitrairement les contributions individuelles ;
- ne pas générer le `CITATION.cff` final avant validation des auteurs, de leur ordre
  et de leurs rôles ;
- ne pas redistribuer logos, poster, photos, génomes ou embeddings sans vérifier
  les droits applicables.

## 7. Organisation du dépôt

```text
1-Introduction/        matériel pédagogique et contexte biologique historique
2-data/                construction du dataset et instructions pour les embeddings
day1/                  baselines k-mers et CNN one-hot
day2/                  teacher sur embeddings Evo2
day3/                  distillation et expériences avancées
day4/                  compromis précision / taille / latence
day5/                  piste SAE exploratoire
src/bioai/             package public et logique réutilisable
tests/                 tests unitaires et de compatibilité
configs/               configurations versionnées
docs/                  méthodologie, résultats, limites, cards et release
examples/              exemples minimaux d’utilisation
reports/               rapports générés localement, sauf artefacts explicitement versionnés
```

Les dossiers `dayN/` sont aussi des archives pédagogiques. Ne pas les réorganiser
massivement ou effacer leurs sorties sans une tâche explicite et une justification.
Les notebooks exploratoires doivent être clairement marqués comme tels plutôt que
présentés comme pipeline canonique.

## 8. Environnement et commandes

Python `3.11+` est la cible principale.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,dev]"
```

Commandes usuelles :

```bash
make lint       # Ruff lint + vérification du format
make test       # Pytest + couverture
make smoke      # smoke tests CLI
make audit      # audit train/validation
make format     # corrections Ruff locales
make serve      # API FastAPI, checkpoint requis via l’environnement
```

Commandes CLI importantes :

```bash
python -m bioai about
python -m bioai features --sequence ATGATGATGATG --output /tmp/features.npy
python -m bioai audit --train TRAIN.csv --evaluation VAL.csv --output audit.json
python -m bioai evaluate --checkpoint MODEL.pt --train TRAIN.csv --evaluation VAL.csv --output metrics.json
python -m bioai predict --checkpoint MODEL.pt --sequence ATG... --device cpu
```

## 9. Règles de développement

### Avant de coder

1. lire les fichiers de référence pertinents ;
2. vérifier `git status` et la branche ;
3. créer une branche dédiée, sauf instruction explicite contraire ;
4. identifier si la tâche touche aux données, aux métriques, au contrat de features
   ou au format de checkpoint ;
5. écrire le test ou le critère de validation avant une refactorisation risquée.

### Pendant la modification

- privilégier de petites fonctions pures et typées ;
- garder la logique scientifique dans `src/bioai`, pas seulement dans un notebook ;
- utiliser des seeds explicites pour les expériences ;
- ne pas masquer les avertissements ou exceptions sans justification ;
- ne pas introduire de dépendance lourde sans expliquer son coût ;
- ne pas renommer les colonnes biologiques ou labels sans migration documentée ;
- éviter les changements de style massifs non liés à la tâche.

### Tests obligatoires

Toute modification de code doit, selon son périmètre, couvrir :

- cas nominal ;
- cas invalides ;
- déterminisme ;
- dimensions et types ;
- compatibilité checkpoint/features ;
- absence de fuite ou surface d’évaluation correcte ;
- compatibilité avec l’implémentation historique lorsque pertinente.

Avant une PR :

```bash
make lint
make test
make smoke
```

La CI doit rester verte. Ne pas réduire le seuil de couverture pour faire passer une
PR.

## 10. Règles pour les modèles et checkpoints

- ne pas hardcoder un seuil dans plusieurs endroits ; le seuil vient de la config
  ou du checkpoint ;
- ne pas charger un checkpoint dont la version de features est incompatible ;
- sauvegarder les métadonnées minimales : seed, commit, config, données/split,
  époque, seuil et métriques de sélection ;
- ne pas publier un checkpoint sélectionné sur la validation finale sans le dire ;
- préférer `state_dict` + configuration explicite à la sérialisation opaque d’un
  objet Python complet ;
- tester le rechargement et une prédiction avant de publier un artefact.

Le checkpoint final OOF reste à retrouver ou régénérer. Ne pas fabriquer un modèle
aléatoire et l’étiqueter comme checkpoint final.

## 11. Règles pour les notebooks

Un notebook destiné à être présenté comme reproductible doit :

- s’exécuter de haut en bas dans un environnement neuf ;
- ne contenir aucune cellule en erreur ou marqueur de crash ;
- fixer les seeds ;
- ne pas dépendre d’un état manuel invisible ;
- annoncer exactement le nombre d’exemples réellement chargé ;
- séparer exploration, sélection et évaluation ;
- enregistrer les résultats utiles dans des artefacts JSON/CSV, pas uniquement dans
  l’affichage de cellules ;
- éviter les titres qui revendiquent un score différent de la sortie exécutée.

Ne pas modifier les sorties pour donner l’impression qu’une exécution a réussi.
Réexécuter réellement ou documenter l’échec.

## 12. Definition of Done

Une tâche est terminée seulement si :

- le comportement demandé est implémenté ;
- les tests pertinents existent et passent ;
- Ruff passe ;
- la documentation et les configs concernées sont mises à jour ;
- aucun secret ni gros artefact indésirable n’est ajouté ;
- les résultats sont formulés sans surinterprétation ;
- l’attribution est conservée ;
- le résumé de PR indique les commandes exécutées et les limites restantes.

Pour une tâche scientifique, ajouter également :

- protocole et surface d’évaluation ;
- seed(s) ;
- commit ;
- artefact des métriques ;
- comparaison équitable avec le témoin ;
- limites et incertitude.

## 13. Blocages connus avant `v1.0.0`

Ne pas les déclarer résolus sans preuve :

1. retrouver ou régénérer le checkpoint student OOF final ;
2. calculer les métriques sur les 993 validations canoniquement inédites ;
3. réexécuter proprement les notebooks principaux ;
4. compléter provenance, licences et checksums ;
5. valider les contributions et l’authorship ;
6. répéter hard vs KD sur davantage de seeds/splits et quantifier l’incertitude ;
7. geler un pipeline unique avant d’ouvrir le test ;
8. créer ensuite seulement la release et les publications finales.

## 14. Priorité recommandée pour un nouvel agent

Sauf instruction différente, traiter dans cet ordre :

1. checkpoint final et export versionné ;
2. évaluation 1 000 / 993 inédites ;
3. nettoyage et réexécution des notebooks canoniques ;
4. provenance et checksums ;
5. attribution collective ;
6. répétitions statistiques et ablations ;
7. release/démonstration.

Lorsqu’une information manque, ne pas deviner : laisser un `TODO` explicite,
ouvrir une issue ou demander la décision humaine nécessaire.
