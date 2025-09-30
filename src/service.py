from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager #
import pickle # 
from pydantic_settings import BaseSettings #
from prometheus_fastapi_instrumentator import Instrumentator 

# ----------------------------------------------------
# A. DÉFINITION DU SCHÉMA DE DONNÉES (PYDANTIC)
# ----------------------------------------------------
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            ]
        }
    }

# ----------------------------------------------------
# B. FONCTION DE CHARGEMENT DES MODÈLES (LIFESPAN)
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage (Application Startup)
    print("--- DÉMARRAGE : Chargement des modèles ML... ---")
    
    # 1. Charger les fichiers .pkl
    with open("logistic_regression.pkl", "rb") as f:
        logreg_model = pickle.load(f)

    with open("random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)

    # 2. Stocker les modèles dans l'état de l'application
    app.state.models = {
        "logreg": logreg_model,
        "random_forest": rf_model,
    }

    
    print("--- DÉMARRAGE : Modèles chargés et prêts. ---")
    yield
    
    # Arrêt (Application Shutdown)
    print("--- ARRÊT : Nettoyage des ressources... ---")
    app.state.models = {}


# ----------------------------------------------------
# C. DÉFINITION DE LA CONFIGURATION (Pydantic Settings)
# ----------------------------------------------------
class Settings(BaseSettings):
    # Ces valeurs seront prioritaires sur les variables d'environnement (API_TITLE, etc.)
    api_title: str = "MLOps Iris Classifier (Default)"
    api_version: str = "2.0.0-DEFAULT"
    api_description: str = "Service with dynamic configuration."
    
    # Permet de charger un fichier .env localement si on lance Python sans Docker Compose
    class Config:
        env_file = ".env" 

settings = Settings() # Instanciation des paramètres

# ----------------------------------------------------
# D. APPLICATION FASTAPI (Initialisation avec lifespan et Settings)
# ----------------------------------------------------
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    # IMPORTANT : Lier la fonction lifespan à l'application
    lifespan=lifespan 
)

# NOUVEAU : Initialiser l'instrumentation APRES la création de l'objet app
Instrumentator().instrument(app).expose(app) # <<< AJOUTER CETTE LIGNE

# ----------------------------------------------------
# E. ENDPOINTS DE STATUS
# ----------------------------------------------------
@app.get("/", tags=["Status"])
async def root():
    """Returns a simple welcome message."""
    return {"message": "Hello World"}

@app.get("/health", tags=["Status"])
async def health_check():
    """Returns the status of the API."""
    return {"status": "healthy"}

# ----------------------------------------------------
# F. ENDPOINT DE PRÉDICTION
# ----------------------------------------------------
IRIS_CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.post("/predict/{model_name}", tags=["Prediction"])
async def predict(model_name: str, features: IrisFeatures):
    """
    Performs classification on the Iris dataset based on the selected model.
    """
    
    # 1. Validation du nom du modèle
    if model_name not in app.state.models:
        return {"error": f"Model '{model_name}' not found. Available models: logreg, random_forest."}

    # 2. Récupération du modèle chargé
    model = app.state.models[model_name]
    
    # 3. Conversion des données Pydantic en format d'entrée ML (Numpy)
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    # 4. Prédiction
    prediction_index = model.predict(input_data)[0]
    
    # 5. Conversion du résultat numérique en nom de fleur lisible
    predicted_class = IRIS_CLASSES[prediction_index]
    
    # 6. Retour de la réponse
    return {
        "model_used": model_name,
        "predicted_class": predicted_class,
        "input_features": features.model_dump() # Affiche les données que le modèle a reçues
    }