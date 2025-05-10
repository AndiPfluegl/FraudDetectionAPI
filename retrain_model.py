import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import sqlite3

# Config via ENV
TRAIN_DB     = os.environ.get("TRAIN_DB", "data/requests.db")
MODEL_OUT    = "models/rf_model.pkl"
MLFLOW_EXPER = "FraudDetection"
MLFLOW_REG   = "fraud-api"

# Daten aus SQLite laden
conn = sqlite3.connect(TRAIN_DB)
# Annahme: Spalte 'Class' enthält das Label, alle anderen V-Features + Amount_scaled sind Merkmale
df = pd.read_sql_query("SELECT * FROM requests", conn)
conn.close()

FEATURE_COLS = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled"]
X = df[FEATURE_COLS]
y = df["Class"] if "Class" in df.columns else df.iloc[:, -1]

# Split für Training/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# MLflow experiment und Run
mlflow.set_experiment(MLFLOW_EXPER)
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MLFLOW_REG
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
