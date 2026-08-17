# Résultats et niveaux de preuve

Ce document évite de mélanger :

- un résultat réellement exécuté ;
- un résultat dont le protocole a également été audité ;
- une expérience exploratoire ;
- une idée non encore mesurée.

## 1. Niveaux de preuve

| Niveau | Définition |
|---|---|
| **Exécuté** | La métrique apparaît dans une sortie de notebook ou un artefact produit par une exécution réelle. |
| **Audité** | Le protocole, les tailles, les chevauchements et la surface d’évaluation ont également été vérifiés. |
| **Exploratoire** | L’expérience a été exécutée, mais un choix a été fait directement sur la validation ou sans protocole OOF complet. |
| **Proposé** | L’idée n’a pas encore été mesurée et ne doit pas être présentée comme un résultat. |

Le split test n’est pas utilisé dans les résultats ci-dessous.

## 2. Parcours pédagogique sur 4 000 exemples

| Étape | Représentation | Modèle | Accuracy | F1 | Paramètres | Preuve |
|---|---|---|---:|---:|---:|---|
| Jour 1 | fréquences de 4-mers | régression logistique | 78,85 % | 80,80 % | 257 | exécuté |
| Jour 1 | one-hot `(4, 200)` | CNN 1D | 83,33 % | 84,13 % | 10 465 | exécuté |
| Jour 2 | embedding Evo2 4096-D | tête MLP | 95,80 % | 95,85 % | 524 545 | exécuté |
| Jour 3 initial | 4-mers | student MLP distillé | ≈ 84,00 % | ≈ 84,31 % | 16 513 | exécuté |

### Interprétation

Le teacher ne correspond pas à l’exécution complète d’Evo2 dans ce dépôt. Les
embeddings Evo2 sont précalculés et Evo2 reste gelé ; les 524 545 paramètres sont
ceux de la tête MLP entraînable.

Le student initial est très petit, mais sa représentation 4-mers ne capture pas
suffisamment la structure biologique pour récupérer l’essentiel de la performance
du teacher.

## 3. Distillation OOF finale, même student avec et sans KD

Les hyperparamètres sont choisis sur un holdout interne groupé par organisme avant
l’ouverture de la validation officielle. Le student utilise 761 caractéristiques
codon/phase et possède 48 961 paramètres.

### Résultats par seed

| Seed | Hard accuracy | KD accuracy | Hard F1 | KD F1 | Hard AUC | KD AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 89,20 % | 90,20 % | 89,62 % | 90,56 % | 95,35 % | 96,43 % |
| 42 | 89,80 % | 90,30 % | 90,19 % | 90,68 % | 95,30 % | 96,21 % |
| 123 | 90,40 % | 89,90 % | 90,73 % | 90,32 % | 95,25 % | 96,24 % |

### Synthèse

| Modèle | Accuracy moyenne | Écart-type accuracy | F1 moyen | AUC moyenne | Preuve |
|---|---:|---:|---:|---:|---|
| MLP codon/phase sans KD | 89,80 % | 0,49 point | 90,18 % | 95,30 % | exécuté |
| MLP codon/phase avec KD OOF | **90,13 %** | **0,17 point** | **90,52 %** | **96,29 %** | exécuté, protocole OOF audité |

### Gain moyen attribuable à la KD dans cette comparaison

| Métrique | Gain KD |
|---|---:|
| Accuracy | +0,33 point |
| F1 | +0,34 point |
| ROC-AUC | +0,99 point |

Le niveau global de performance dépend fortement des 761 caractéristiques. La
formulation correcte est donc :

> Le passage du student initial à environ 84 % vers 90,13 % vient principalement
d’une meilleure représentation biologique, complétée par une distillation OOF.
La comparaison avec le même student sans KD isole un gain moyen de 0,33 point
d’accuracy et de 0,99 point d’AUC attribuable aux cibles teacher dans le protocole
actuel.

### Limite de la surface d’évaluation

L’audit du sous-ensemble 4 000/1 000 a identifié 7 validations ayant un équivalent
exact ou reverse-complement dans le train. Le notebook final rapporte actuellement
les métriques sur les 1 000 lignes. Il doit encore produire les métriques séparées
sur les **993 fenêtres canoniques inédites**.

Le statut de 90,13 % est donc **exécuté et méthodologiquement amélioré**, mais la
release `v1.0` devra ajouter cette seconde surface d’évaluation.

## 4. Expériences supervisées sur toutes les données

Ces expériences utilisent jusqu’à 58 550 exemples train et 10 373 validations.
Elles ne sont pas directement comparables au défi volontaire des 4 000 embeddings.

| Modèle | Accuracy | F1 | AUC | Surface | Preuve |
|---|---:|---:|---:|---|---|
| Régression logistique enrichie | 92,26 % | 92,31 % | 97,44 % | validation complète | exécuté |
| HistGradientBoosting, 37 résumés | 95,34 % | 95,42 % | 98,48 % | validation complète | exécuté |
| ExtraTrees, seed 42 | 95,85 % | 95,94 % | 98,80 % | validation complète avant retrait des overlaps | exécuté |
| ExtraTrees, seed 42 | **95,47 %** | **95,86 %** | **98,57 %** | 9 501 fenêtres canoniques inédites | **audité** |
| Vote ET/HGB/LR | 95,82 % | — | — | validation inédite | exploratoire |

Le vote est exploratoire parce que les poids `0,60 / 0,20 / 0,20` ont été examinés
directement sur la validation, sans apprentissage OOF de l’ensemble.

## 5. CNN 1D multi-échelle

| Mesure | Valeur |
|---|---:|
| Paramètres | 116 169 |
| Meilleure époque | 13 |
| Époques exécutées | 19, avec early stopping |
| Accuracy | 93,12 % |
| F1 | 93,22 % |
| Temps local total | 202,3 s |
| Preuve | exécuté, validation complète |

Ce modèle dépasse le CNN simple du Jour 1, mais reste derrière les meilleurs
modèles utilisant les caractéristiques codon/phase.

## 6. Distillation hybride sur toutes les données

Les logits teacher étaient disponibles seulement pour 4 000 exemples sur les
58 550, soit environ 6,83 % du train.

| Configuration | Accuracy | F1 | Écart vs hard |
|---|---:|---:|---:|
| Hard, sans KD | 95,09 % | 95,10 % | référence |
| KD `alpha=0,98`, `T=2` | 95,04 % | 95,04 % | -0,06 point |
| KD `alpha=0,98`, `T=4` | 94,97 % | 94,97 % | -0,13 point |
| KD `alpha=0,50`, `T=4` | 94,15 % | 94,19 % | -0,94 point |

Aucune variante n’a battu le modèle hard-only. Cette expérience a motivé la
production de logits OOF plutôt que des cibles teacher surconfiantes issues
d’exemples vus.

## 7. Efficacité du student final

| Modèle | Accuracy | Paramètres | Taille | Features | Forward CPU | Total |
|---|---:|---:|---:|---:|---:|---:|
| Student initial 4-mers | 84,00 % | 16 513 | 0,0684 Mo | 0,0399 ms | 0,0124 ms | 0,0523 ms |
| Student codon/phase + KD OOF | 90,13 % | 48 961 | 0,1986 Mo | 0,3879 ms | 0,0190 ms | 0,4069 ms |

Le student final utilise environ 2,96 fois plus de paramètres et 7,78 fois plus de
temps total que le student initial, pour un gain de 6,13 points d’accuracy.

Environ 95,3 % du temps total vient de l’extraction des caractéristiques. La
priorité d’optimisation n’est donc pas la couche MLP, mais la transformation des
séquences en 761 features.

Ces latences sont des médianes locales et ne doivent pas être généralisées sans
préciser le matériel et l’environnement.

## 8. Résultats non validés

### Autoencodeur parcimonieux

Le notebook bonus propose d’analyser des activations Evo2 avec un autoencodeur
Top-K afin de rechercher des features séparant codant et non codant et une
périodicité de trois nucléotides.

Dans l’état actuel, cette piste n’a pas été exécutée de bout en bout. Elle est donc
**proposée**, pas démontrée.

### Notebook `03_test_distillation_experiments.ipynb`

Son titre et certains commentaires annoncent des niveaux ou tailles de données qui
ne correspondent pas aux sorties actuellement enregistrées. Il ne doit pas servir
de preuve avant nettoyage et réexécution.

## 9. Formulations sûres

### À utiliser

> Avec 4 000 exemples, un student MLP de 48 961 paramètres et une distillation
> out-of-fold groupée par organisme, nous obtenons 90,13 % d’accuracy moyenne sur
> trois seeds. Le modèle n’utilise plus Evo2 à l’inférence.

> Pour le même student et les mêmes features, les cibles teacher OOF apportent en
> moyenne +0,33 point d’accuracy et +0,99 point d’AUC par rapport à l’entraînement
> hard-only.

> Sur toutes les données supervisées, ExtraTrees avec les caractéristiques
> codon/phase atteint 95,47 % sur les fenêtres de validation sans équivalent exact
> ou reverse-complement dans le train.

### À éviter

- « Le student a compressé tout Evo2 » ;
- « La distillation seule nous a fait passer de 84 % à 90 % » ;
- « Le student dépasse toujours 90 % » ;
- « 95,82 % est notre score final validé » ;
- « Le test confirme les résultats » ;
- « Tous les notebooks sont intégralement reproductibles dans l’état actuel ».
