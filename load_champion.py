# ===== 1. Imports =====
import mlflow
import mlflow.sklearn  # nécessaire pour recharger un modèle scikit-learn
import pandas as pd


# ===== 2. Connexion au tracking server (même base que le TP3) =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Chargement du modèle champion via son alias =====
# On cible l'alias plutôt qu'une version précise : on ne connaît
# ni le numéro de version ni le run_id sous-jacent.
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print(f"Modèle chargé : {type(model).__name__}")
print(f"Variables attendues : {list(model.feature_names_in_)}")


# ===== 4. Construction d'un nouveau client fictif =====
# Les colonnes d'abonnement sont déjà encodées en 0/1 :
# ici le client a un abonnement Standard (et pas Premium).
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])


# ===== 5. Alignement des colonnes avec celles attendues par le modèle =====
# On ne conserve que les colonnes connues du modèle, dans le bon ordre,
# pour éviter toute erreur de schéma au moment de la prédiction.
client = client[list(model.feature_names_in_)]


# ===== 6. Prédiction : 1 = départ, 0 = resté =====
prediction = model.predict(client)[0]
print(f"Le client va-t-il partir ? → {'OUI' if prediction == 1 else 'NON'}")