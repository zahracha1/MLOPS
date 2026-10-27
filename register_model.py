# ===== 1. Imports =====
import mlflow
from mlflow import MlflowClient


# ===== 2. Connexion au tracking server (même base que le TP3) =====
# On pointe vers le fichier SQLite local qui contient tous nos runs
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Récupération du meilleur run de l'expérience =====
# On trie les runs par F1-score décroissant : le premier est le meilleur
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"],
)
best_run = runs.iloc[0]
run_id = best_run["run_id"]

print(f"Meilleur run : {best_run['tags.mlflow.runName']} | id : {run_id[:8]}...")


# ===== 4. Enregistrement du modèle dans le Model Registry =====
# "runs:/<run_id>/model" cible le sous-dossier 'model' du run sélectionné
# "churn-model" est le nom du modèle dans le catalogue (créé si absent)
registered = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-model",
)
print(f"Modèle enregistré : churn-model | version {registered.version}")


# ===== 5. Attribution de l'alias 'champion' à cette version =====
client = MlflowClient()
client.set_registered_model_alias(
    name="churn-model",
    alias="champion",
    version=registered.version,
)
print(f"Alias 'champion' → version {registered.version}")