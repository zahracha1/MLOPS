import pandas as pd  # Construire et manipuler des tableaux de données
import numpy as np   # Générer des valeurs numériques aléatoires

# Fixer la graine aléatoire pour garantir la reproductibilité
# des expériences : les mêmes données seront recréées à chaque exécution.
np.random.seed(42)

# Nombre de clients fictifs à générer
n = 500

# Construire le tableau de données colonne par colonne.
# Chaque colonne représente une FEATURE (donnée d'entrée du futur modèle).
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),  # Âge entre 18 et 69 ans
    "salaire": np.random.randint(1200, 8000, n),  # Salaire mensuel simulé
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"], n
    ),  # Type d'offre
    "anciennete": np.random.randint(0, 15, n),  # Ancienneté en années
    "consommation": np.random.randint(
        0, 100, n
    ),  # Score d'utilisation du service, sur 100
})

# Créer la TARGET (colonne à prédire) : churn.
# Règle simplifiée et volontairement arbitraire :
# un client peu actif (consommation < 30) ET récent
# (ancienneté < 2 ans) est considéré comme à risque de départ.

data["churn"] = (
    (data["consommation"] < 30)
    & (data["anciennete"] < 2)
).astype(int)  # Convertir True/False en 1/0

# Sauvegarder le tableau dans un fichier CSV.
# Attention : ce fichier ne bénéficie d'aucun système de versionnement.
data.to_csv("churn_data.csv", index=False)

# Afficher les cinq premières lignes pour vérifier le résultat.
print(data.head())

# Afficher le taux de churn en pourcentage.
print(f"Taux de churn : {data['churn'].mean():.1%}")  