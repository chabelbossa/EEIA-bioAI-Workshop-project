# Checklist de publication

## Bloquants scientifiques

- [ ] Produire les métriques du student final sur les 993 validations canoniques inédites.
- [ ] Répéter hard vs KD sur davantage de seeds ou splits groupés.
- [ ] Ajouter intervalles de confiance et métriques de calibration.
- [ ] Choisir un seul pipeline avant toute ouverture du test.
- [ ] Documenter le protocole exact de l'unique évaluation test finale.

## Bloquants de provenance

- [ ] Confirmer la source, la version et la licence de chaque FASTA/GFF.
- [ ] Produire un manifeste SHA256 des données et embeddings.
- [ ] Confirmer les conditions de redistribution des embeddings Evo2.
- [ ] Valider les droits sur le poster, les logos et photographies.

## Bloquants d'attribution

- [ ] Faire confirmer les noms et orthographes par les participants.
- [ ] Remplir le tableau de contributions dans `CONTRIBUTORS.md`.
- [ ] Distinguer auteurs, contributeurs et superviseurs.
- [ ] Transformer `CITATION.cff.template` en `CITATION.cff` seulement après validation.

## Qualité logicielle

- [x] Package `src/bioai` et contrat de features versionné.
- [x] Tests unitaires du featurizer, de la KD, des audits et des checkpoints.
- [x] CI Ruff + Pytest + smoke test sans embeddings.
- [x] CLI d'audit et d'inférence.
- [x] Service FastAPI et Dockerfile optionnels.
- [ ] Verrouiller les versions dans un lock généré et testé sur Python 3.11.
- [ ] Réexécuter proprement tous les notebooks principaux.
- [ ] Nettoyer/archiver les notebooks exploratoires incohérents.
- [ ] Publier un checkpoint final versionné.

## Release

- [ ] Mettre à jour `CHANGELOG.md`.
- [ ] Ajouter les figures finales et le poster autorisé sous `docs/assets/`.
- [ ] Créer une release candidate.
- [ ] Faire relire méthodologie, attribution et claims.
- [ ] Créer `v1.0.0` et, si pertinent, archiver sur Zenodo.
- [ ] Publier ensuite le post LinkedIn technique en reprenant uniquement les formulations validées.
