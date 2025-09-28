from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager # Import nécessaire pour lifespan
import pickle # Import nécessaire pour charger les modèles

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
# Le décorateur asynccontextmanager transforme la fonction en un contexte de cycle de vie.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage (Application Startup)
    # C'est ici que les modèles sont chargés UNE SEULE FOIS.
    print("--- DÉMARRAGE : Chargement des modèles ML... ---")
    
    # 1. Charger les fichiers .pkl
    with open("logistic_regression.pkl", "rb") as f:
        logreg_model = pickle.load(f)

    with open("random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)

    # 2. Stocker les modèles dans l'état de l'application
    # app.state est l'endroit idéal pour stocker des ressources globales.
    app.state.models = {
        "logreg": logreg_model,
        "random_forest": rf_model,
    }
    
    print("--- DÉMARRAGE : Modèles chargés et prêts. ---")
    yield
    
    # Arrêt (Application Shutdown)
    # Optionnel, mais bonne pratique pour les grosses ressources.
    print("--- ARRÊT : Nettoyage des ressources... ---")
    app.state.models = {}


# ----------------------------------------------------
# C. APPLICATION FASTAPI (Initialisation avec lifespan)
# ----------------------------------------------------
app = FastAPI(
    title="MLOps FastAPI Service",
    description="A service for Iris classification.",
    version="1.0.0",
    # IMPORTANT : Lier la fonction lifespan à l'application
    lifespan=lifespan 
)

# ... Les endpoints / et /health restent ici ...
@app.get("/", tags=["Status"])
async def root():
    """Returns a simple welcome message."""
    return {"message": "Hello World"}

@app.get("/health", tags=["Status"])
async def health_check():
    """
    Returns the status of the API.
    """
    return {"status": "healthy"}

# NOTE : L'endpoint de prédiction vient à l'étape suivante !


# ----------------------------------------------------
# D. ENDPOINT DE PRÉDICTION
# ----------------------------------------------------

# Mapping des indices de classe aux noms de fleurs pour un résultat lisible
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
    # On crée une liste des valeurs [sepal_length, sepal_width, ...]
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    # 4. Prédiction
    # model.predict prend une liste de listes (ou un tableau Numpy)
    prediction_index = model.predict(input_data)[0]
    
    # 5. Conversion du résultat numérique en nom de fleur lisible
    predicted_class = IRIS_CLASSES[prediction_index]
    
    # 6. Retour de la réponse
    return {
        "model_used": model_name,
        "predicted_class": predicted_class,
        "input_features": features.model_dump() # Affiche les données que le modèle a reçues
    }