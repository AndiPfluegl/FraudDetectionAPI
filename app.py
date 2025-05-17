import os
import logging
from flask import Flask, request, jsonify, Response
import pickle
import numpy as np
import sqlite3
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps

# Configuration: environment variables with defaults
API_TOKEN = os.environ.get("API_TOKEN") or (
    lambda: (_ for _ in ()).throw(
        RuntimeError("Environment variable API_TOKEN must be set")
    )
)()
MODEL_PATH = os.environ.get("MODEL_PATH", "models/rf_model.pkl")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", 0.4))
LATEST_DB = os.environ.get("LATEST_DB", "data/requests.db")

# Ensure the database directory exists
os.makedirs(os.path.dirname(LATEST_DB), exist_ok=True)

# Initialize Flask application and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SQLite database with required schema if not exists
logger.info("Initializing SQLite DB at %s", LATEST_DB)
def init_db():
    conn = sqlite3.connect(LATEST_DB)
    # Define feature columns V1..V28 and Amount_scaled
    feature_cols = ", ".join(f"V{i} REAL" for i in range(1, 29)) + ", Amount_scaled REAL"
    # Create requests table with score and timestamp
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS requests (
            {feature_cols},
            score REAL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

# Decorator to require a valid API token in Authorization header
def require_token(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapped

# Prometheus metrics setup
metrics = PrometheusMetrics(app, path=None)
PREDICT_COUNTER = Counter(
    "predictions_total", "Number of predictions executed", ["predicted_class"]
)
FRAUD_PROB = Histogram(
    "fraud_probabilities", "Distribution of fraud probabilities",
    buckets=[i / 10 for i in range(11)]
)
metrics.info("app_info", "Fraud Detection Service Info", version="1.0")

# Load the trained model from file
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None

# Feature names expected by the model
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]

# Health check endpoint (optional)
@app.route("/health", methods=["GET"])
@require_token
def health():
    return jsonify({"status": "ok"}), 200

# Metrics endpoint
@app.route("/metrics", methods=["GET"])
@require_token
def metrics_endpoint():
    data = generate_latest(metrics.registry)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)

# Predict endpoint: accepts JSON with 2D list "data" and returns fraud probabilities
@app.route("/predict", methods=["POST"])
@require_token
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Parse input
    content = request.get_json(force=True)
    data = content.get("data")
    if data is None:
        return jsonify({"error": "No 'data' field in JSON"}), 400

    arr = np.array(data)
    # Validate shape
    if arr.ndim != 2 or arr.shape[1] != len(FEATURES):
        return jsonify({
            "error": f"Expected 2D array with {len(FEATURES)} features, got {arr.shape}"
        }), 400

    # Compute fraud probabilities
    probs = model.predict_proba(arr)[:, 1]

    # Record Prometheus metrics
    for p in probs:
        label = "fraud" if p >= FRAUD_THRESHOLD else "legit"
        PREDICT_COUNTER.labels(predicted_class=label).inc()
        FRAUD_PROB.observe(p)

    # Persist input features and score to SQLite
    columns = FEATURES + ["score"]
    placeholders = ",".join("?" for _ in columns)
    insert_sql = f"INSERT INTO requests ({','.join(columns)}) VALUES ({placeholders})"
    rows = [list(row) + [float(score)] for row, score in zip(arr.tolist(), probs.tolist())]

    with sqlite3.connect(LATEST_DB) as conn:
        conn.executemany(insert_sql, rows)

    # Return probabilities
    return jsonify({"fraud_probability": probs.tolist()})

# Entry point for local debugging
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)