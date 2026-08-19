# Contribuer au projet

## Principes

1. Séparer clairement résultat exécuté, résultat exploratoire et proposition.
2. Ne jamais utiliser le split test pour choisir un modèle ou un hyperparamètre.
3. Grouper les splits par organisme lorsque le protocole l'exige.
4. Auditer les séquences exactes et reverse-complements avant de publier une métrique.
5. Conserver les en-têtes d'attribution du matériel pédagogique amont.
6. Ne pas attribuer une contribution sans confirmation factuelle.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,dev]"
pre-commit install
```

## Vérifications obligatoires

```bash
make lint
make test
make smoke
```

La CI n'utilise ni embeddings Evo2 ni données volumineuses. Les tests doivent donc rester déterministes et pouvoir s'exécuter sur un petit runner CPU.

## Convention de branches et commits

- branches : `feat/...`, `fix/...`, `docs/...`, `experiment/...` ;
- commits : Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`) ;
- une PR doit indiquer sa surface d'évaluation et si le test a été consulté.

## Ajouter une expérience

Une expérience rapportable doit enregistrer :

- commit Git ;
- configuration et seed ;
- données et IDs utilisés ;
- groupe de split ;
- métriques et seuil ;
- matériel et versions ;
- artefacts produits ;
- limites et décisions prises à partir du résultat.

Toute sélection faite directement sur la validation officielle doit être signalée comme exploratoire.

## Modifier les caractéristiques

`FEATURE_VERSION` et `FEATURE_DIM` forment un contrat public. Toute modification de l'ordre ou du calcul des 761 valeurs exige :

- une nouvelle version de features ;
- un test de migration ;
- de nouveaux checkpoints ;
- une mise à jour de la model card.

## Données et secrets

- ne jamais committer de clé NVIDIA ou autre secret ;
- ne pas ajouter de fichier volumineux sans stratégie DVC/LFS/Hugging Face/Zenodo ;
- ne pas republier des données externes sans licence vérifiée ;
- préférer des fixtures synthétiques dans les tests.
