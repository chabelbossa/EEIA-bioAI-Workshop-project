# Limites et menaces à la validité

Cette page fait partie du résultat. Elle précise ce que les expériences montrent,
ce qu’elles ne montrent pas et les contrôles encore nécessaires avant une release
scientifiquement défendable.

## 1. Intitulé exact du problème

Le problème réellement traité est :

> classer des fenêtres d’ADN bactérien de 200 nucléotides en **codantes** ou
> **non codantes**.

Ce n’est pas, dans l’état actuel, une classification de la **fonction** des
protéines. Détecter une région codante et attribuer une fonction biologique à la
protéine correspondante sont deux tâches différentes.

## 2. Chevauchements entre train et validation

Même lorsque les organismes sont disjoints, des séquences identiques peuvent
apparaître dans plusieurs génomes.

L’audit actuel a trouvé, sur la validation complète :

- 848 séquences uniques exactes communes aux splits ;
- 863 séquences uniques communes après prise en compte du reverse-complement ;
- 872 lignes de validation concernées.

ExtraTrees obtient 100 % sur ce sous-ensemble, ce qui est compatible avec une
mémorisation de motifs identiques. Le score prudent est donc rapporté sur 9 501
fenêtres canoniques inédites.

Sur le sous-ensemble Evo2 4 000/1 000, 7 validations ont un équivalent exact ou
reverse-complement dans le train. Le score du student final sur les 993 validations
inédites reste à calculer.

### Action exigée

Toute table finale devra présenter au minimum :

1. le score sur toute la validation officielle ;
2. le score sur la validation canonique inédite ;
3. le nombre d’exemples retirés et la règle de canonicalisation.

## 3. Similarité biologique au-delà des doublons exacts

Retirer les doublons exacts et reverse-complements ne suffit pas à garantir une
vraie nouveauté biologique. Deux fenêtres peuvent rester très homologues sans être
identiques.

Le split par organisme limite une partie du problème, mais ne garantit pas une
séparation par espèce, genre, famille ou clade.

### Actions recommandées

- estimer la similarité séquentielle entre splits ;
- regrouper les organismes taxonomiquement ;
- tester un split par clade ou un seuil de similarité ;
- documenter la taxonomie et la provenance de chaque génome.

## 4. Asymétrie d’orientation dans la construction du dataset

Les fenêtres positives du brin négatif sont reverse-complémentées pour être placées
dans le sens de lecture. Les fenêtres négatives restent dans l’orientation de
référence et reçoivent `strand = "+"`.

Cette décision peut :

- aider à exprimer la structure réelle des codons ;
- introduire un raccourci artificiel entre orientation et label ;
- rendre la prédiction sensible au choix arbitraire du brin de référence.

### Expériences nécessaires

- augmentation aléatoire par reverse-complement dans les deux classes ;
- canonicalisation des deux classes ;
- moyenne des prédictions `p(seq)` et `p(RC(seq))` ;
- mesure de l’écart de prédiction entre les deux orientations ;
- ablation des features direct/reverse.

## 5. Effet propre de la distillation

Le student final atteint 90,13 % d’accuracy moyenne avec KD contre 89,80 % sans KD.
Le gain moyen actuel est donc de 0,33 point, pas de 6,13 points.

Le passage d’environ 84 % à 90,13 % provient surtout du remplacement des seuls
4-mers par 761 caractéristiques biologiques plus riches.

### Limites statistiques

- seulement trois seeds ;
- un seul holdout interne pour le réglage ;
- validation officielle de 1 000 exemples ;
- seuil de décision différent de 0,5 ;
- absence actuelle d’intervalle de confiance du différentiel hard/KD.

### Actions nécessaires

- répéter les splits groupés et les seeds ;
- utiliser une comparaison appariée ;
- ajouter bootstrap et intervalles de confiance ;
- rapporter le gain absolu avec son incertitude ;
- tester la calibration et la robustesse au seuil.

## 6. Teacher et notion de compression

Le teacher utilisé dans les notebooks est une tête MLP entraînée sur des embeddings
Evo2 précalculés. Evo2 lui-même n’est pas entraîné ni embarqué dans ce dépôt.

Le student apprend une tâche spécialisée à partir de labels durs et de probabilités
teacher. Il ne reproduit pas l’ensemble des capacités d’Evo2.

### Formulation interdite

> « Nous avons compressé Evo2 dans un MLP de 48 961 paramètres. »

### Formulation correcte

> « Nous avons distillé, pour une tâche binaire spécialisée, des cibles produites
> par une tête entraînée sur les représentations Evo2 vers un student autonome. »

## 7. Taille et représentativité du sous-ensemble Evo2

Le protocole principal utilise 4 000 exemples train et 1 000 validations. Cette
contrainte est utile pour comparer les représentations, mais limite la portée des
conclusions.

Il faut notamment vérifier :

- la représentativité des organismes sélectionnés ;
- la distribution des longueurs et des bases ambiguës ;
- l’équilibre par organisme, et pas uniquement par classe ;
- la stabilité sur des organismes très différents du train.

## 8. Jeu équilibré et conditions réelles

Les classes sont équilibrées artificiellement par organisme. Cette configuration
facilite l’apprentissage et rend l’accuracy lisible, mais la proportion réelle de
régions codantes et intergéniques peut être différente dans un génome complet.

Conséquences :

- la probabilité prédite n’est pas automatiquement calibrée pour une prévalence
  réelle différente ;
- accuracy et F1 sur un jeu équilibré ne suffisent pas pour un usage de criblage ;
- précision et rappel dépendront du seuil et du contexte de déploiement.

Il faudra évaluer le modèle sur une distribution naturelle avant de parler
d’annotation automatique d’un génome complet.

## 9. Définition des négatifs

Les négatifs sont extraits des régions intergéniques. Ils ne représentent pas
nécessairement tous les cas difficiles :

- pseudogènes ;
- petits ORFs ;
- régions régulatrices ;
- séquences chevauchantes ;
- erreurs d’annotation ;
- fragments de gènes ;
- ARN non codants.

La tâche actuelle dépend donc fortement de la qualité des annotations GFF et de la
construction des négatifs.

## 10. Qualité des annotations

Les labels ne sont pas des vérités absolues : ils proviennent d’annotations
biologiques pouvant contenir des omissions ou erreurs.

Avant une publication scientifique, il faut documenter :

- la source exacte de chaque FASTA/GFF ;
- la version des annotations ;
- la date de téléchargement ;
- les licences ;
- les critères de filtrage ;
- les éventuelles annotations « predicted » ou « hypothetical ».

## 11. Fenêtres fixes et frontières de gènes

Une fenêtre positive est extraite à l’intérieur d’une `CDS`. Le modèle n’est donc
pas nécessairement entraîné à détecter précisément le début ou la fin d’un gène.

Il répond plutôt à :

> « Cette fenêtre ressemble-t-elle à une portion interne de région codante ? »

Il ne faut pas confondre cette tâche avec :

- la segmentation complète d’un génome ;
- la localisation exacte des coordonnées d’un gène ;
- la prédiction du cadre de lecture ;
- la fonction de la protéine.

## 12. Métriques manquantes

Les métriques principales sont accuracy, F1 et ROC-AUC. Pour une analyse plus
complète, il manque encore :

- précision et rappel ;
- PR-AUC ;
- matrice de confusion agrégée et par organisme ;
- sensibilité/spécificité ;
- Brier score ;
- courbe de calibration ;
- intervalles de confiance ;
- robustesse au seuil ;
- performance par organisme et par contenu GC.

## 13. Comparaison des latences

Les latences actuelles sont mesurées localement et ne sont pas directement
comparables à l’inférence complète d’Evo2, puisque les embeddings teacher sont
précalculés.

Le tableau d’efficacité compare surtout :

- les students autonomes ;
- la tête MLP du teacher à partir d’un embedding déjà disponible.

Pour une comparaison système complète, il faudrait inclure :

- coût d’extraction Evo2 ;
- mémoire du modèle de fondation ;
- matériel ;
- batch size ;
- temps de chargement ;
- latence p50/p95 ;
- débit ;
- coût énergétique ou financier.

## 14. Performance du feature engineering

Pour le student final, environ 95,3 % du temps total vient de l’extraction des 761
features. Le réseau n’est donc pas le goulot d’étranglement.

Les mesures actuelles proviennent d’une implémentation Python/NumPy lisible, pas
optimisée. Une vectorisation ou une implémentation compilée pourrait modifier
fortement la comparaison de latence.

## 15. Reproductibilité

Le dépôt n’est pas encore une release entièrement reproductible :

- dépendances non épinglées ;
- absence de fichier de lock ;
- absence de tests et CI ;
- gros embeddings téléchargés séparément sans checksums ;
- certains notebooks partiellement exécutés ;
- deux notebooks avancés conservent un marqueur de crash ;
- matériel exact de benchmark non documenté automatiquement.

Voir [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## 16. Piste SAE non validée

L’autoencodeur parcimonieux est une piste d’interprétabilité, pas un résultat du
pipeline final. Le notebook n’est pas exécuté de bout en bout et aucune conclusion
biologique ne doit être tirée de cette section dans l’état actuel.

## 17. Usage responsable

Le projet est éducatif et expérimental. Il ne doit pas être utilisé comme outil de
diagnostic médical, de décision clinique ou d’annotation de référence sans :

- validation externe ;
- expertise bioinformatique ;
- comparaison à des outils établis ;
- traçabilité des données ;
- contrôle humain ;
- analyse des erreurs et de l’incertitude.
