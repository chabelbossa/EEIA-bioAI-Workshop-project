# Politique de sécurité

## Signaler un problème

Ne publiez pas dans une issue publique :

- une clé d'API ;
- une donnée biologique confidentielle ;
- un chemin contenant des informations personnelles ;
- une vulnérabilité exploitable du service de démonstration.

Contactez d'abord le mainteneur du dépôt par un canal privé GitHub approprié.

## Secrets

Le dépôt ignore `.env` et les noms contenant `api_key`. Les notebooks et logs doivent être inspectés avant chaque commit, car une sortie Jupyter peut conserver une valeur secrète même si le code source ne la contient plus.

Après exposition d'un secret :

1. le révoquer immédiatement ;
2. générer une nouvelle clé ;
3. nettoyer l'historique si nécessaire ;
4. vérifier les forks, artefacts CI et caches.

## Service d'inférence

L'API fournie est une démonstration, sans authentification ni rate limiting. Elle ne doit pas être exposée sur Internet telle quelle. Avant déploiement :

- limiter la longueur des séquences ;
- imposer quotas et timeout ;
- éviter de journaliser les séquences ;
- épingler et scanner les dépendances ;
- exécuter le conteneur sans privilèges ;
- valider l'origine et le hash du checkpoint.
