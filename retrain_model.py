import os
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow

# Configuration
TRAIN_DB        = os.environ.get("TRAIN_DB", "data/requests.db")
MODEL_OUT       = "models/rf_model.pkl"
MLFLOW_EXPER    = os.environ.get("MLFLOW_EXPER", "FraudDetection")
MLFLOW_REG      = os.environ.get("MLFLOW_REG", "fraud-api")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", 0.5))

# Load data from SQLite
conn = sqlite3.connect(TRAIN_DB)
# requests table should contain V1..V28, Amount_scaled, score, ts
df = pd.read_sql_query("SELECT * FROM requests", conn)
conn.close()

# Features
feature_cols = [col for col in df.columns if col.startswith("V")] + ["Amount_scaled"]
X = df[feature_cols]

# Labels: use actual Class if available, otherwise derive from score threshold
if "Class" in df.columns:
    y = df["Class"]
else:
    if "score" not in df.columns:
        raise ValueError("No 'score' column found to derive labels from.")
    y = (df["score"] >= FRAUD_THRESHOLD).astype(int)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# MLflow experiment & run
mlflow.set_experiment(MLFLOW_EXPER)
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Log accuracy
    acc = clf.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)

    # Register the model
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        registered_model_name=MLFLOW_REG
    )

    # Save locally
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
