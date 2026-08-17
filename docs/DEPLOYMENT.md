# Déploiement du student compact

## État actuel

Le dépôt fournit une interface d'inférence et un service FastAPI optionnel. Aucun checkpoint final n'est encore publié dans une release : le service doit donc recevoir un checkpoint versionné produit avec `bioai.model.save_checkpoint`.

## 1. CLI

```bash
pip install -e .

bioai predict \
  --checkpoint artifacts/student-oof.pt \
  --sequence ATGCGT... \
  --device cpu
```

La sortie contient la probabilité `coding`, la classe, le seuil, la version de features et les métadonnées du checkpoint.

## 2. API locale

```bash
pip install -e ".[serve]"
export BIOAI_CHECKPOINT=/chemin/student-oof.pt
uvicorn bioai.api:app --host 0.0.0.0 --port 8000
```

Endpoints :

- `GET /health` : état du modèle et version de features ;
- `POST /predict` : prédiction sur une séquence ADN.

Exemple :

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"ATGATGATGATG"}'
```

## 3. Docker

```bash
docker build -t eeia-bioai:0.1.0 .

docker run --rm -p 8000:8000 \
  -e BIOAI_CHECKPOINT=/models/student.pt \
  -v "$PWD/artifacts:/models:ro" \
  eeia-bioai:0.1.0
```

## 4. Contrat de checkpoint

Un artefact public doit contenir :

- `format_version` ;
- `feature_version = codon-phase-761-v1` ;
- `model_config` ;
- `threshold` ;
- `state_dict` ;
- `metadata` : seed, date, commit, split, métriques et matériel.

Le chargement échoue explicitement si la version de features ne correspond pas.

## 5. Points à compléter avant un déploiement public

- publier le checkpoint final choisi après gel du pipeline ;
- ajouter son SHA256 et sa licence ;
- tester l'image Docker en CI ;
- ajouter limites de taille et rate limiting ;
- journaliser la version du modèle sans conserver les séquences sensibles ;
- ajouter un exemple d'entrée non sensible ;
- afficher un avertissement « usage recherche/éducation, non clinique » ;
- publier des métriques de test uniquement après décision collective d'ouvrir le split.
