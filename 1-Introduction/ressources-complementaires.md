<h1 align="center">📚 Ressources complémentaires</h1>

<p align="center">
  <img src="https://img.shields.io/badge/NIVEAU-Débutant%20en%20biologie-5eead4?style=for-the-badge&labelColor=0d1526" alt="Niveau"/>
  <img src="https://img.shields.io/badge/AVANT-Le%20Jour%201-818cf8?style=for-the-badge&labelColor=0d1526" alt="Quand"/>
</p>

> **À qui s'adresse cette page ?**
> Vous venez de l'informatique, des maths ou de la data science, pas de la biologie.
> Cette page rassemble le strict minimum à comprendre **avant** d'ouvrir le premier
> notebook. Rien ici n'est un prérequis noté : c'est du contexte pour que les données
> aient du sens.
>
> Le déroulé de la semaine reste dans [`guide.md`](guide.md).

---

## 🎬 1. Comprendre l'ADN en vidéo

Commencez par là. Une vidéo vaut mieux qu'un long paragraphe pour se faire une
première image de ce qu'est l'ADN, un gène, et pourquoi on peut le traiter comme
du texte.

<div align="center">

[![Introduction à l'ADN](https://img.youtube.com/vi/2JUu1WqidC4/maxresdefault.jpg)](https://youtu.be/2JUu1WqidC4?si=ZiZ2AGcRCbo_GqnY)

**▶️ [Regarder la vidéo](https://youtu.be/2JUu1WqidC4?si=ZiZ2AGcRCbo_GqnY)**

</div>

**Ce qu'il faut en retenir pour l'atelier :**

| Notion | En une phrase | Pourquoi ça compte ici |
|---|---|---|
| **ADN** | Une longue chaîne de 4 lettres : `A`, `C`, `G`, `T` | C'est littéralement notre entrée : une chaîne de caractères |
| **Génome** | L'ADN complet d'un organisme | Nos bactéries font ~2 à 6 millions de lettres |
| **Gène / CDS** | Un segment qui code une protéine | C'est la classe `1` que l'on cherche à prédire |
| **Intergénique** | Ce qu'il y a *entre* les gènes | C'est la classe `0` |

> 💡 **L'idée centrale de la semaine :** si l'ADN est du texte, alors tout ce que
> l'on sait faire en NLP (n-grammes, embeddings, modèles de fondation, distillation)
> s'y applique. La tâche « codant vs non-codant » est notre classification binaire.

---

## 🗄️ 2. Le NCBI : d'où viennent nos données

### C'est quoi ?

Le **NCBI** (*National Center for Biotechnology Information*) est l'institut public
américain qui héberge les grandes bases de données de la biologie. C'est
**le dépôt de référence mondial** pour les séquences biologiques : quasiment tout
génome publié y est déposé, gratuitement et publiquement accessible.

🔗 **<https://www.ncbi.nlm.nih.gov/>**

Une analogie qui marche bien : le NCBI est à la biologie ce que **GitHub est au
code**, ou ce que **Hugging Face est aux modèles**. Un dépôt central, versionné,
avec des identifiants stables et une API.

### Les briques utiles à connaître

| Ressource | Contenu | Analogie |
|---|---|---|
| **GenBank** | Toutes les séquences soumises par les chercheurs | Le dépôt brut, tel que soumis |
| **RefSeq** | Une sélection curée et non redondante | La branche « stable », relue |
| **Assembly** | Les génomes complets assemblés | Une *release* taguée |
| **SRA** | Les lectures brutes de séquençage | Les données avant traitement |

### Lire un identifiant

Nos fichiers portent des noms comme celui-ci :

```text
GCA_002442855.1_ASM244285v1.fasta
└┬┘ └───┬────┘ ┬  └────┬─────┘
 │      │      │       └─ nom de l'assemblage donné par le soumetteur
 │      │      └───────── version (.1, .2, …)
 │      └──────────────── numéro d'accession, unique et permanent
 └─────────────────────── GCA = assemblage GenBank   (GCF = version RefSeq)
```

**Le point important :** un accession est **stable dans le temps**. `GCA_002442855.1`
désignera toujours exactement le même assemblage, partout dans le monde. C'est ce qui
rend un travail reproductible — exactement comme un commit SHA.

### Les deux formats de fichiers de l'atelier

Chaque génome arrive en deux fichiers de même nom de base :

**`.fasta` — la séquence elle-même**

```text
>CP003195.1 Propionibacterium acnes TypeIA2 P.acn33, complete genome
CTAGCGTTCAGGGGAGTGGTACATCGTGGGTAGCTTGTCACACCGACTGTGGAAAACTGT
GTGGACAACTCTCGACAACTTCTAGAGAACAGGTGGTGGAATGTCCGACACACCGTTCGG
...
```

Une ligne d'en-tête commençant par `>`, puis la séquence sur les lignes suivantes.
C'est tout. Le format date de 1985 et n'a pas bougé.

**`.gff` — les annotations**

Un tableau tabulé qui dit *où* se trouvent les gènes dans la séquence :

```text
CP003195.1  Genbank  CDS  101   1603  .  +  0  ID=cds-AEW78118.1;product=...
CP003195.1  Genbank  CDS  1991  3235  .  +  0  ID=cds-AEW78119.1;product=...
└────┬───┘           └┬┘  └─┬─┘ └─┬─┘    ┬
  séquence          type  début  fin   brin (+ ou −)
```

> 🔑 **C'est de là que viennent nos étiquettes.** Une fenêtre de 200 lettres qui
> tombe dans un intervalle `CDS` reçoit le label **1** (codant) ; sinon **0**
> (non-codant). Tout le travail de `2-data/build_dataset.py` tient dans cette phrase.

### Aller voir par soi-même

1. Ouvrez <https://www.ncbi.nlm.nih.gov/datasets/genome/>
2. Cherchez un organisme (`Escherichia coli`, par exemple)
3. Choisissez un assemblage, puis **Download** → cochez *Genomic sequence (FASTA)*
   et *Annotation features (GFF3)*

Vous obtenez exactement la paire de fichiers présente dans [`../2-data/raw/`](../2-data/raw/).

> ℹ️ Vous n'avez **rien à télécharger** pour l'atelier : les génomes sont déjà dans
> le dépôt. Cette section est là pour que vous sachiez d'où ils sortent et comment
> en récupérer d'autres si vous voulez prolonger le projet.

---

## 🧭 3. Mini-glossaire

| Terme | Définition courte |
|---|---|
| **Base / nucléotide** | Une lettre : `A`, `C`, `G` ou `T` |
| **pb** (paire de bases) | L'unité de longueur. Nos fenêtres font 200 pb |
| **Brin** (`+` / `−`) | L'ADN a deux brins lus en sens opposés |
| **CDS** | *Coding Sequence* — un segment traduit en protéine |
| **Intergénique** | Région située entre deux gènes |
| **Codon** | Un triplet de bases. La périodicité 3 du Jour 5 vient de là |
| **Assemblage** | Un génome reconstitué à partir de fragments séquencés |
| **Annotation** | Les métadonnées de position : où sont les gènes |
| **FASTA / GFF** | Les deux formats ci-dessus : séquence / annotations |

---

## ➕ 4. Autres ressources

<!-- Ajoutez ici les ressources supplémentaires au fil de la semaine.
     Format suggéré, pour rester lisible :

### 📄 Titre de la ressource
**Type :** article · vidéo · outil · cours
🔗 <https://…>
Une ou deux phrases : ce que ça apporte et à quel moment de la semaine le lire.
-->

*Cette section sera enrichie au fil de la semaine.*

---

<div align="center">

**Prêt·e ?** → [`guide.md`](guide.md) → [`../day1/`](../day1/)

</div>
