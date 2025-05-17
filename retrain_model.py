"""
Retrain Model Script

This script loads labeled transaction data from a SQLite database, performs a stratified
train/test split, and retrains a RandomForestClassifier. The new model is logged to MLflow
and saved locally as a Pickle file.
"""
import os
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow

# --- Configuration ---
# Path to SQLite database containing 'requests' table
TRAIN_DB = os.environ.get("TRAIN_DB", "data/requests.db")
# Output path for the retrained model
MODEL_OUT = os.environ.get("MODEL_OUT", "models/rf_model.pkl")
# MLflow experiment and registered model names
MLFLOW_EXPER = os.environ.get("MLFLOW_EXPER", "FraudDetection")
MLFLOW_REG = os.environ.get("MLFLOW_REG", "fraud-api")
# Threshold to convert scores into binary class if no 'Class' column
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", 0.5))

# --- Load data from SQLite ---
print(f"Loading data from database: {TRAIN_DB}")
conn = sqlite3.connect(TRAIN_DB)
df = pd.read_sql_query("SELECT * FROM requests", conn)
conn.close()
print(f"Total rows in 'requests' table: {len(df)}")

# --- Filter rows that have a label: either 'Class' or 'score' ---
has_class = df.get("Class").notna() if "Class" in df else pd.Series(False, index=df.index)
has_score = df.get("score").notna()
df = df[has_class | has_score]
print(f"Rows with valid labels (Class or score): {len(df)}")

# --- Prepare features (V1..V28 + Amount_scaled) ---
feature_cols = [col for col in df.columns if col.startswith("V")] + ["Amount_scaled"]
X = df[feature_cols]

# --- Determine target labels y ---
if "Class" in df.columns:
    # Use 'Class' when available, otherwise derive from 'score'
    y = df["Class"].where(df["Class"].notna(),
                            (df["score"] >= FRAUD_THRESHOLD).astype(int))
else:
    # No 'Class' column: require 'score' for binary labels
    if "score" not in df.columns:
        raise ValueError("No 'score' column found to derive labels from.")
    y = (df["score"] >= FRAUD_THRESHOLD).astype(int)

# --- Ensure stratification is possible ---
unique_classes = len(pd.unique(y))
if unique_classes < 2:
    raise ValueError(
        "Not enough class diversity for stratified split. Check your score thresholds or data."
    )
print(f"Unique label classes for stratification: {unique_classes}")

# --- Split into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- MLflow experiment and run setup ---
mlflow.set_experiment(MLFLOW_EXPER)
with mlflow.start_run():
    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate and log accuracy
    accuracy = clf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Logged accuracy: {accuracy}")

    # Log model to MLflow registry and save locally
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        registered_model_name=MLFLOW_REG
    )
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"Model saved to: {MODEL_OUT}")
