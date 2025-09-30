
# 🤖 Service ML d'Iris - Déploiement MLOps Complet

Ce projet démontre le déploiement d'un service de prédiction de Machine Learning (modèle Iris) à l'aide d'un stack MLOps moderne et conteneurisé. L'objectif est de garantir l'**observabilité** et la **flexibilité** en production.

## 🌟 Caractéristiques Principales

  * **API ML Productionnelle :** Service de prédiction haute performance construit avec **FastAPI**.
  * **Conteneurisation :** Isolation et reproductibilité de l'environnement grâce à **Docker**.
  * **Orchestration :** Déploiement multi-services (API, Prometheus, Grafana) géré par **Docker Compose**.
  * **Configuration Sécurisée :** Utilisation de **Pydantic Settings** et du fichier **`.env`** pour externaliser les variables de configuration (version, secrets) de l'image Docker.
  * **Observabilité / Monitoring :** Intégration complète de **Prometheus** (collecte de métriques) et **Grafana** (visualisation) pour surveiller la santé et les performances de l'API en temps réel.

-----

## 🛠️ Stack Technique

| Composant | Rôle |
| :--- | :--- |
| **Framework API** | FastAPI / Uvicorn |
| **Conteneurisation** | Docker |
| **Orchestration** | Docker Compose |
| **Modèles** | Scikit-learn (fichiers `.pkl`) |
| **Monitoring** | Prometheus |
| **Visualisation** | Grafana |
| **Gestion Config** | Pydantic Settings / `.env` |

-----

## 🚀 Démarrage Rapide

Ces instructions vous permettent de construire et de lancer l'intégralité du stack sur votre machine.

### 1\. Prérequis

Assurez-vous d'avoir installé sur votre système :

  * **Docker Desktop** (recommandé, avec WSL activé si sur Windows).
  * **Git**.

### 2\. Construction de l'Image Docker

Nous construisons l'image de l'API en incluant l'instrumentation Prometheus.

```bash
# Construire l'image de l'API ML (version 3.1)
docker build -t fastapi-ml-service:v3.1 .
```

### 3\. Lancement du Stack Complet

Cette commande lit le `docker-compose.yml`, démarre l'API, Prometheus et Grafana, et attache le fichier de configuration `.env`.

```bash
# Lancer l'API, Prometheus et Grafana en mode détaché (-d)
docker compose up -d
```

-----

## 🔗 Endpoints et Configuration

Une fois le stack lancé (cela peut prendre quelques secondes pour que l'API démarre), les services sont accessibles via les adresses suivantes :

| Service | Adresse | Détails |
| :--- | :--- | :--- |
| **API ML** | `http://localhost:8000/docs` | Documentation interactive (Swagger UI) pour les endpoints de prédiction. |
| **Métriques API** | `http://localhost:8000/metrics` | Endpoint des métriques au format Prometheus. |
| **Prometheus** | `http://localhost:9090` | Interface de consultation et vérification des cibles. |
| **Grafana** | `http://localhost:3000` | Tableaux de bord de visualisation. |

### Configuration de la Source de Données Grafana

1.  Accédez à **Grafana** (`http://localhost:3000`) et connectez-vous (**admin / admin**).
2.  Allez dans **Connections** \> **Data Sources** et ajoutez une source de données **Prometheus**.
3.  Dans le champ **URL**, utilisez l'adresse interne pour la communication entre conteneurs :
    $$\text{http://prometheus:9090}$$
    *(Si cette adresse échoue, essayez l'adresse de l'hôte Docker : `http://host.docker.internal:9090`)*.

-----

## 🧹 Nettoyage

Pour arrêter et supprimer tous les conteneurs et le réseau créé par Docker Compose :

```bash
# Arrêter et supprimer les conteneurs
docker compose down
```