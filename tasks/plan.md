# Plan d'expérimentation Day 3

## Objectif

Comparer sur le même split train/validation plusieurs représentations et stratégies
d'apprentissage capables de dépasser l'ensemble actuel à 90,91 %, sans utiliser le
split test pour choisir les hyperparamètres.

## Expériences parallèles

1. Caractéristiques biologiques orientées phase/codons et modèle tabulaire.
2. CNN 1D sur la séquence one-hot, afin de conserver l'ordre des nucléotides.
3. Distillation hybride : labels durs sur tout le train et logits Evo2 sur les
   exemples alignés disponibles.
4. Amélioration de l'ensemble par pondération/stacking sans fuite de validation.

## Critères d'acceptation

- Chaque essai utilise les 58 552 lignes train et les 10 374 lignes validation,
  hors séquences dont la longueur n'est pas 200.
- Les graines et métriques Accuracy/F1 sont rapportées.
- Le test reste inutilisé pendant la sélection.
- Seules les variantes réellement exécutées sont présentées comme mesurées.

## Risques

- Les logits Evo2 ne couvrent que 4 000 exemples : la distillation doit être masquée.
- Une pondération choisie directement sur validation peut surapprendre : privilégier
  les prédictions out-of-fold ou présenter l'essai comme exploratoire.
- Les temps CPU peuvent limiter le nombre de graines et d'hyperparamètres.
