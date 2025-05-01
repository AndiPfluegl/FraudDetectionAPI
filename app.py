import csv
import os
import logging
from flask import Flask, request, jsonify, Response
import pickle
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
from datetime import datetime

# ——— Konfiguration ———
API_TOKEN        = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Environment variable API_TOKEN must be set")
MODEL_PATH       = os.environ.get("MODEL_PATH", "models/rf_model.pkl")
FRAUD_THRESHOLD  = float(os.environ.get("FRAUD_THRESHOLD", 0.4))
LATEST_DATA_PATH = os.environ.get("LATEST_DATA_PATH", "data/latest_data.csv")

# ——— Verzeichnis & Header für latest_data.csv anlegen ———
os.makedirs(os.path.dirname(LATEST_DATA_PATH), exist_ok=True)
# FEATURES kennen wir erst nach dem Laden des Modells, darum hier Platzhalter
# Wir erzeugen den Header erst weiter unten, sobald FEATURES definiert ist.

# ——— App & Logging ———
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Einfacher Token-Check Decorator ———
def require_token(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapped

# ——— Metrics Setup (ohne automatischen /metrics) ———
metrics = PrometheusMetrics(app, path=None)

@app.route("/metrics")
@require_token
def metrics_endpoint():
    data = generate_latest(metrics.registry)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)

metrics.info("app_info", "Fraud Detection Service Info", version="1.0")

# ——— Modell-Metriken ———
PREDICT_COUNTER = Counter(
    "predictions_total",
    "Number of predictions",
    ["predicted_class"]
)
FRAUD_PROB = Histogram(
    "fraud_probabilities",
    "Distribution of predicted fraud probability",
    buckets=[i/10 for i in range(11)]
)

# ——— Modell laden ———
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None

# ——— FEATURES definieren und Header in latest_data.csv schreiben ———
FEATURES = ["V" + str(i) for i in range(1, 29)] + ["Amount_scaled"]
# Header einfügen, falls noch nicht geschehen
if os.path.isfile(LATEST_DATA_PATH):
    # Datei existiert, aber prüfen, ob leer oder nicht-CSV-Header
    with open(LATEST_DATA_PATH, "r", newline="") as f:
        first = f.readline().strip().split(",")
    if first != FEATURES:
        # Datei ist leer oder fehlerhafter Header: neu anlegen
        with open(LATEST_DATA_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURES)
else:
    # Datei fehlt ganz
    with open(LATEST_DATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURES)

# ——— Predict Endpoint ———
@app.route("/predict", methods=["POST"])
@require_token
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    content = request.get_json(force=True)
    data = content.get("data")
    if data is None:
        return jsonify({"error": "No 'data' field in JSON"}), 400

    input_data = np.array(data)
    if input_data.ndim != 2 or input_data.shape[1] != len(FEATURES):
        return jsonify({
            "error": f"Expected 2D array with {len(FEATURES)} features, got {input_data.shape}"
        }), 400

    probs = model.predict_proba(input_data)[:, 1]

    # Metriken aktualisieren
    for p in probs:
        label = "fraud" if p >= FRAUD_THRESHOLD else "legit"
        PREDICT_COUNTER.labels(predicted_class=label).inc()
        FRAUD_PROB.observe(p)

    # ==== Logging der Features ins latest_data.csv ====
    try:
        with open(LATEST_DATA_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            for row in input_data.tolist():
                writer.writerow(row)
    except Exception as e:
        logger.error("Could not write to latest_data.csv: %s", e)
    # ===================================================

    return jsonify({"fraud_probability": probs.tolist()})

# ——— App starten ———
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
