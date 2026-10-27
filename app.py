# ===== 1. Imports =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Création de l'application FastAPI =====
# Le titre apparaît dans la page de documentation interactive /docs
app = FastAPI(title="API Churn")


# ===== 3. Chargement du modèle champion (une seule fois, au démarrage) =====
# Recharger le modèle à chaque requête rendrait l'API très lente,
# donc on le charge une seule fois en mémoire au lancement.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Description du schéma d'entrée (validé automatiquement par Pydantic) =====
class Client(BaseModel):
    age: int                                          # nombre entier
    salaire: float                                    # nombre décimal
    abonnement: Literal["Basique", "Standard", "Premium"]  # 3 valeurs autorisées
    anciennete: float
    consommation: float


# ===== 5. Route /health : vérifier que l'API est opérationnelle =====
@app.get("/health")  # GET : lecture seule, aucun corps envoyé
def health() -> dict:
    return {"status": "ok"}


# ===== 6. Route /predict : recevoir un client, renvoyer la prédiction =====
@app.post("/predict")  # POST : on envoie un client dans le corps
def predict(client: Client) -> dict:
    # a) Transformation du client reçu en DataFrame d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # b) Encodage one-hot de la colonne 'abonnement' (comme à l'entraînement)
    ligne = pd.get_dummies(ligne)

    # c) Alignement sur les colonnes attendues par le modèle
    #    (colonnes manquantes comblées par 0, ordre respecté)
    ligne = ligne.reindex(columns=model.feature_names_in_, fill_value=0)

    # d) Prédiction : 1 = départ, 0 = resté
    resultat = int(model.predict(ligne)[0])
    message = "Le client va partir" if resultat == 1 else "Le client reste"

    return {"churn": resultat, "message": message}