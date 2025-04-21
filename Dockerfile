# 1. Basis‑Image
FROM python:3.10-slim AS builder

# 2. Arbeitsverzeichnis
WORKDIR /app

# 3. System‑Dependencies (falls nötig für Pandas/NumPy)
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# 4. Requirements kopieren und installieren
COPY requirements.txt .
# gleich MLflow, SciPy und Joblib für Retrain/Drift mitinstallieren
RUN pip install --no-cache-dir -r requirements.txt scipy joblib mlflow

# 5. Alles in den finalen Container kopieren
FROM python:3.10-slim
WORKDIR /app

# nutze die bereits installierten Pakete
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# kopiere Service‑Code, Modell und MLOps‑Skripte
COPY app.py \
     rf_model.pkl \
     retrain_model.py \
     drift_detector.py \
     drift_config.json \
     ./

# 6. Port freigeben
EXPOSE 5000

# 7. Produktion: Gunicorn als WSGI‑Server
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]
