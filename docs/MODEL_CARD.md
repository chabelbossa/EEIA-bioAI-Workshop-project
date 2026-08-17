# Model card — student MLP codon/phase + distillation OOF

## 1. Résumé

Le modèle principal est un classifieur binaire compact qui reçoit 761 caractéristiques calculées directement depuis une fenêtre d'ADN et produit un logit codant / non codant.

Il a été développé pour répondre à la question suivante : une partie de l'information portée par les embeddings Evo2 peut-elle améliorer un petit modèle autonome, sans appeler Evo2 à l'inférence et sans utiliser des cibles teacher produites sur des exemples déjà vus ?

## 2. Architecture

```text
761 entrées
  -> Linear(761, 64)
  -> LayerNorm(64)
  -> GELU
  -> Dropout(0,2)
  -> Linear(64, 1)
```

Nombre exact de paramètres entraînables : **48 961**.

Le format de checkpoint public enregistre :

- l'architecture ;
- la version des caractéristiques ;
- le seuil de décision ;
- le `state_dict` ;
- des métadonnées d'entraînement.

Les `state_dict` bruts sans contrat de features sont volontairement refusés par l'API publique.

## 3. Entrées

Le contrat `codon-phase-761-v1` contient :

| Bloc | Dimensions |
|---|---:|
| k-mers globaux, k=1..4 | 340 |
| codons par phase, brin direct | 192 |
| codons par phase, reverse-complement | 192 |
| résumés start/stop/segments sans stop | 24 |
| fréquences A/C/G/T selon la phase | 12 |
| dispersion GC entre phases | 1 |
| **Total** | **761** |

L'implémentation publique est testée contre le module historique afin de détecter toute dérive silencieuse de l'ordre des features.

## 4. Teacher et distillation

Le teacher utilise des embeddings Evo2 de dimension 4 096 et une tête MLP :

```text
4096 -> Linear(4096, 128) -> ReLU -> Dropout(0,1) -> Linear(128, 1)
```

Pour produire les cibles de distillation :

1. les données sont séparées par organisme ;
2. cinq teachers temporaires sont entraînés avec `GroupKFold` ;
3. chaque logit est produit par un teacher qui n'a vu ni l'exemple ni son organisme ;
4. le student apprend avec une combinaison de labels réels et de logits OOF.

Configuration sélectionnée dans le notebook exécuté :

- `alpha = 0.5` ;
- `temperature = 1` ;
- 57 époques ;
- seuil `0.265` ;
- seeds finales : 7, 42 et 123.

## 5. Résultats validés à ce stade

### Student, sous-ensemble 4 000 / 1 000

| Modèle | Accuracy moyenne | F1 moyen | ROC-AUC moyenne | Écart-type accuracy |
|---|---:|---:|---:|---:|
| Même architecture, hard labels uniquement | 89,80 % | 90,18 % | 95,30 % | 0,49 point |
| Student avec distillation OOF | **90,13 %** | **90,52 %** | **96,29 %** | **0,17 point** |

Gain moyen observé de la KD, à architecture et features identiques :

- +0,33 point d'accuracy ;
- +0,34 point de F1 ;
- +0,99 point de ROC-AUC ;
- variance plus faible entre les trois seeds.

Ces chiffres sont des résultats de validation, pas des métriques de test final. Deux seeds KD dépassent 90 %, la troisième atteint 89,9 %. Il faut donc dire **90,13 % en moyenne**, et non « toujours plus de 90 % ».

### Teacher

Le teacher Evo2 gelé + tête MLP atteint 95,80 % d'accuracy et 95,85 % de F1 sur la surface documentée du parcours. Ce score ne signifie pas que le student a compressé l'intégralité d'Evo2.

## 6. Taille et latence

Mesures locales CPU rapportées :

| Élément | Valeur |
|---|---:|
| Paramètres student | 48 961 |
| Checkpoint sérialisé | environ 0,1986 Mo |
| Extraction des 761 features | 0,3879 ms / séquence |
| Forward MLP | 0,0190 ms / séquence |
| Total | 0,4069 ms / séquence |

Environ 95,3 % du temps total vient de l'extraction des caractéristiques. L'optimisation prioritaire concerne donc le featurizer plutôt que le réseau.

Ces latences dépendent du matériel, des versions logicielles et de la méthodologie de mesure.

## 7. Usage prévu

Usage acceptable :

- apprentissage et démonstration de baselines génomiques ;
- recherche exploratoire sur la distillation et les représentations ;
- comparaison de protocoles anti-fuite ;
- démonstration d'un pipeline ML reproductible.

Usage non validé :

- annotation clinique ;
- diagnostic ;
- prise de décision biomédicale ;
- annotation de production sans validation indépendante ;
- généralisation hors du domaine microbien de l'atelier.

## 8. Limites

- seulement trois seeds dans la comparaison finale ;
- gain KD modeste et significativité statistique non établie ;
- performance globale fortement due aux 761 features biologiques ;
- 7 chevauchements canoniques dans la validation 1 000 ;
- métrique spécifique sur les 993 fenêtres inédites encore à produire ;
- pas de score test final ouvert ;
- pas de split par homologie/clade ;
- calibration et PR-AUC à ajouter au rapport final ;
- checkpoint public final non encore attaché à une release.

## 9. Reproductibilité

Le modèle public est défini dans `src/bioai/model.py`, la perte dans `src/bioai/distillation.py`, les features dans `src/bioai/features.py` et la configuration reconstruite dans `configs/student_oof.json`.

```bash
pip install -e ".[dev]"
pytest --cov=bioai
bioai about
```

L'exécution complète de la distillation reste documentée dans `day3/03_serious_oof_distillation.ipynb` jusqu'à extraction totale de l'entraînement dans le package.
