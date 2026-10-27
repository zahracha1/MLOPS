# ===== 1. Imports =====
import requests  # client HTTP minimal pour interroger l'API


# ===== 2. Configuration =====
URL = "http://localhost:8000"  # adresse locale de l'API Churn


# ===== 3. Vérification de l'état de l'API =====
response = requests.get(f"{URL}/health")
print(f"GET /health → {response.json()}")


# ===== 4. Client « à risque » : faible consommation, faible ancienneté =====
client_1 = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10,
}
# json= sérialise automatiquement le dictionnaire en JSON et
# positionne le header Content-Type: application/json
response = requests.post(f"{URL}/predict", json=client_1)
print(f"POST /predict (client 1) → {response.status_code} {response.json()}")


# ===== 5. Client « fidèle » : forte consommation, forte ancienneté =====
client_2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80,
}
response = requests.post(f"{URL}/predict", json=client_2)
print(f"POST /predict (client 2) → {response.status_code} {response.json()}")


# ===== 6. Cas d'erreur volontaire : champ 'age' en texte =====
# dict(...) crée une copie de client_1 avec 'age' remplacé par "abc"
mauvais_client = {**client_1, "age": "abc"}
response = requests.post(f"{URL}/predict", json=mauvais_client)

# FastAPI/Pydantic renvoie un code 422 (Unprocessable Entity) :
# le corps contient la liste 'detail' des erreurs de validation.
print(f"POST /predict (age='abc') → {response.status_code}")
print(f"Message : {response.json()['detail'][0]['msg']}")