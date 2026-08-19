# Data card — fenêtres génomiques codantes / non codantes

## 1. Objet du jeu de données

Le jeu de données sert à une tâche de classification binaire sur des fenêtres d'ADN bactérien de 200 nucléotides :

- `label = 1` : fenêtre extraite à l'intérieur d'une région `CDS` annotée ;
- `label = 0` : fenêtre extraite d'une région intergénique, hors des `CDS`.

Il ne s'agit pas d'un jeu de données de classification des **fonctions de protéines**. Le problème effectivement implémenté est la détection de fenêtres codantes.

## 2. Construction

Le script [`2-data/build_dataset.py`](../2-data/build_dataset.py) associe, pour chaque organisme :

- un fichier FASTA contenant la séquence génomique ;
- un fichier GFF contenant les annotations, dont les lignes `CDS`.

Les coordonnées GFF sont converties en intervalles Python 0-based, puis :

1. les fenêtres positives sont extraites à l'intérieur des `CDS` ;
2. les fenêtres négatives sont extraites dans le complément intergénique ;
3. les séquences du brin négatif sont converties en reverse-complement ;
4. les fenêtres contenant plus de 5 % de `N` sont rejetées ;
5. les deux classes sont équilibrées par organisme ;
6. les lignes sont mélangées avec une seed explicite.

La commande historique utilise `--window 200 --stride 200`. Avec ces paramètres, les fenêtres d'un même intervalle ne se chevauchent pas. Une autre valeur de stride modifierait cette propriété.

## 3. Schéma des CSV

| Colonne | Signification |
|---|---|
| `id` | identifiant de la fenêtre dans le split |
| `split` | `train`, `val` ou `test` |
| `organism` | identifiant de l'organisme source |
| `seqid` | séquence/contig source dans le FASTA |
| `start`, `end` | coordonnées de la fenêtre |
| `strand` | orientation de la région codante ; `+` pour les négatifs |
| `label` | 1 codant, 0 non codant |
| `sequence` | chaîne ADN normalisée |

## 4. Volumes observés dans le dépôt

| Split | Lignes brutes | Usage actuel |
|---|---:|---|
| Train | 58 552 | entraînement et expériences internes |
| Validation | 10 374 | comparaison des modèles ; jamais utilisée comme test |
| Test | 18 448 | volontairement fermé pendant la sélection |

Trois lignes ne respectaient pas la longueur attendue de 200 nucléotides : deux dans le train et une dans la validation. Les expériences avancées sur toutes les données utilisent donc 58 550 lignes train et 10 373 lignes validation après ce contrôle.

## 5. Sous-ensemble aligné avec Evo2

Les embeddings Evo2 pré-calculés couvrent :

- 4 000 exemples train, équilibrés 2 000 / 2 000 ;
- 1 000 exemples validation, équilibrés 500 / 500 ;
- un split test conservé hors sélection.

Le sous-échantillonnage est déterministe, équilibré par classe et utilise la seed 42. Il a été retenu pour comparer plusieurs représentations à quantité de données constante.

## 6. Séparation par organisme

Dans les expériences documentées :

- le sous-ensemble train contient 20 organismes ;
- la validation officielle contient 4 organismes distincts ;
- la distillation finale produit les cibles teacher avec `GroupKFold`, en groupant par `organism`.

Cette séparation limite la fuite directe par organisme, mais ne garantit pas à elle seule l'absence de séquences identiques, reverse-complements ou homologues entre splits.

## 7. Audit des chevauchements

Sur la validation complète, l'audit enregistré a identifié :

- 848 séquences uniques exactes présentes dans train et validation ;
- 863 séquences uniques communes après canonicalisation par reverse-complement ;
- 872 lignes de validation concernées ;
- 9 501 fenêtres canoniques réellement inédites.

Sur le sous-ensemble Evo2 4 000 / 1 000 :

- 7 fenêtres de validation possèdent un équivalent exact ou reverse-complement dans le train ;
- 993 fenêtres sont canoniquement inédites.

Le package public fournit :

```bash
bioai audit \
  --train 2-data/processed/train.csv \
  --evaluation 2-data/processed/val.csv \
  --output reports/overlap-audit.json
```

## 8. Risques de biais et limites

1. **Domaine restreint** : les données concernent des génomes microbiens présents dans l'atelier ; elles ne démontrent pas une généralisation à tous les organismes.
2. **Échantillonnage artificiellement équilibré** : la proportion codant/non codant ne reflète pas nécessairement la fréquence réelle d'usage.
3. **Fenêtres locales** : une fenêtre de 200 nt ne représente pas toujours une unité biologique complète.
4. **Doublons et homologie** : l'audit exact/reverse-complement ne remplace pas un split par similarité de séquence ou clade taxonomique.
5. **Annotations sources** : les labels dépendent de la qualité des GFF.
6. **Provenance juridique** : la licence MIT du code ne détermine pas les droits de redistribution des FASTA, GFF et embeddings.
7. **Test fermé** : aucune métrique de test finale n'est revendiquée dans la version actuelle.

## 9. Provenance et licences à compléter avant archivage

Avant une release Zenodo, Hugging Face ou une redistribution indépendante, documenter pour chaque organisme :

- l'identifiant de la source ;
- l'URL ou accession ;
- la version du FASTA et du GFF ;
- la date de téléchargement ;
- la licence ou les conditions de réutilisation ;
- le hash SHA256 des fichiers.

Les embeddings Evo2 doivent également recevoir un manifeste avec dimensions, couche extraite, modèle précis, date, identifiants alignés et checksums.
