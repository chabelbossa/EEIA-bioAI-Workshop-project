# Changelog

Toutes les modifications notables sont documentées ici.

## [Unreleased]

### Added

- README de projet orienté publication et claims prudents ;
- documentation méthodologie, résultats, limites et reproductibilité ;
- package `bioai` avec features codon/phase 761-D ;
- audit de chevauchement exact et reverse-complement ;
- student MLP versionné de 48 961 paramètres ;
- perte de distillation binaire stable ;
- métriques explicites et format de checkpoint ;
- CLI `about`, `features`, `audit` et `predict` ;
- API FastAPI et Dockerfile optionnels ;
- tests unitaires et CI ;
- data card, model card et checklist de publication.

### Changed

- le problème est nommé « détection de régions codantes » plutôt que « classification de fonctions de protéines » ;
- les résultats validés sont séparés des expériences exploratoires ;
- le featurizer public préalloue la matrice et met en cache l'indexation des k-mers tout en conservant le vecteur historique.

### Pending before v1.0.0

- validation des contributions et de la citation ;
- provenance/licences/checksums des données et embeddings ;
- métriques student sur les 993 validations canoniques inédites ;
- réexécution propre des notebooks ;
- checkpoint final et unique ouverture du test.
