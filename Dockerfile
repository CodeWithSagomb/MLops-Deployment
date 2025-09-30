# Utiliser une image de base Python légère (Best Practice: slim-buster)
FROM python:3.11-slim-buster


WORKDIR /app


COPY requirements.txt .

# 
RUN pip install --no-cache-dir -r requirements.txt

# 
# 
COPY logistic_regression.pkl .
COPY random_forest.pkl .

# 
COPY src /app/src/

# --- Best Practice Sécurité
# 
RUN useradd --no-create-home --shell /bin/false mluser

# Changer le propriétaire du répertoire /app à 'mluser'
RUN chown -R mluser:mluser /app

# Passer à l'utilisateur non-root
USER mluser

# Le port utilisé par Uvicorn
EXPOSE 8000

# Commande d'exécution du serveur Uvicorn
# 
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]