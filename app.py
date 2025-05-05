import os
import logging
from flask import Flask, request, jsonify, Response
import pickle
import numpy as np
import sqlite3
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps

# ——— Konfiguration ———
API_TOKEN       = os.environ.get("API_TOKEN") or (lambda: (_ for _ in ()).throw(
                   RuntimeError("Environment variable API_TOKEN must be set")))()
MODEL_PATH      = os.environ.get("MODEL_PATH", "models/rf_model.pkl")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", 0.4))
LATEST_DB       = os.environ.get("LATEST_DB", "data/requests.db")

# ——— Verzeichnisse anlegen ———
os.makedirs(os.path.dirname(LATEST_DB), exist_ok=True)

# (Optional, wenn Du CSV nicht mehr brauchst:)
# LATEST_DATA_PATH = os.environ.get("LATEST_DATA_PATH", "data/latest_data.csv")
# os.makedirs(os.path.dirname(LATEST_DATA_PATH), exist_ok=True)

# ——— App & Logging ———
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— DB initialisieren ———
logger.info("Using LATEST_DB at %s", LATEST_DB)
def init_db():
    conn = sqlite3.connect(LATEST_DB)
    cols = ", ".join(f"V{i} REAL" for i in range(1,29)) + ", Amount_scaled REAL"
    # score und ts ergänzen
    conn.execute(f"""
      CREATE TABLE IF NOT EXISTS requests(
        {cols},
        score REAL,
        ts   DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    """)
    conn.commit()
    conn.close()

init_db()

# ——— Token-Decorator ———
def require_token(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapped

# ——— Metrics Setup ———
metrics = PrometheusMetrics(app, path=None)
@app.route("/metrics")
@require_token
def metrics_endpoint():
    data = generate_latest(metrics.registry)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)
metrics.info("app_info", "Fraud Detection Service Info", version="1.0")

PREDICT_COUNTER = Counter("predictions_total","Number of predictions",["predicted_class"])
FRAUD_PROB       = Histogram("fraud_probabilities","Distribution of predicted fraud probability",
                              buckets=[i/10 for i in range(11)])

# ——— Modell laden ———
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None

FEATURES = ["V"+str(i) for i in range(1,29)] + ["Amount_scaled"]

# ——— Predict Endpoint ———
@app.route("/predict", methods=["POST"])
@require_token
def predict():
    if model is None:
        return jsonify({"error":"Model not loaded"}), 500

    content = request.get_json(force=True)
    data    = content.get("data")
    if data is None:
        return jsonify({"error":"No 'data' field in JSON"}), 400

    arr = np.array(data)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURES):
        return jsonify({
            "error": f"Expected 2D array with {len(FEATURES)} features, got {arr.shape}"
        }), 400

    # Fraud-Probabilities berechnen
    probs = model.predict_proba(arr)[:, 1]

    # Metrics sammeln
    for p in probs:
        lbl = "fraud" if p >= FRAUD_THRESHOLD else "legit"
        PREDICT_COUNTER.labels(predicted_class=lbl).inc()
        FRAUD_PROB.observe(p)

    # Inputs + Score in einem Rutsch in SQLite speichern (ohne ts)
    columns = FEATURES + ["score"]
    placeholders = ",".join("?" for _ in columns)
    col_list = ",".join(columns)

    rows = [row + [p] for row, p in zip(arr.tolist(), probs.tolist())]

    with sqlite3.connect(LATEST_DB) as conn:
        conn.executemany(
            f"INSERT INTO requests ({col_list}) VALUES ({placeholders})",
            rows
        )

    return jsonify({"fraud_probability": probs.tolist()})


if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)
