import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import os

# Config
DATA_PATH     = "data/latest_data.csv"
MODEL_OUT     = "models/rf_model.pkl"
MLFLOW_EXPER  = "FraudDetection"
MLFLOW_REG    = "fraud-api"

# Daten laden
df = pd.read_csv(DATA_PATH)
X = df[[c for c in df.columns if c.startswith("V")] + ["Amount_scaled"]]
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# MLflow experiment starten
mlflow.set_experiment(MLFLOW_EXPER)
with mlflow.start_run():
    # Modell trainieren
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Metriken loggen
    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)

    # Artefakt speichern
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MLFLOW_REG
    )

    # Lokal als Pickle
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
