import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Chargement du dataset Iris (inclus dans scikit-learn)
iris = load_iris()
X, y = iris.data, iris.target
print("Dataset Iris chargé. X.shape:", X.shape)

# 2. Entraînement du modèle de Régression Logistique
# max_iter augmenté pour garantir la convergence (une bonne pratique)
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X, y)
print("Modèle de Régression Logistique entraîné.")

# 3. Entraînement du modèle de Forêt Aléatoire
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
print("Modèle de Forêt Aléatoire entraîné.")

# 4. Sauvegarde des modèles au format .pkl
# Note : Les fichiers PKL seront créés à la racine du projet, à côté du dossier src.
with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(logreg_model, f)
print("Modèle sauvegardé sous : logistic_regression.pkl")

with open("random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Modèle sauvegardé sous : random_forest.pkl")

print("--- Entraînement et sauvegarde terminés ! ---")