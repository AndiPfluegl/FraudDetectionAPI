# 1. Build‑Stage: Alle Dependencies installieren
FROM python:3.10-slim AS builder

# Arbeitsverzeichnis (beliebig)
WORKDIR /app

# System‑Dependencies (für Pandas/NumPy, ggf. auch scipy-Komponenten)
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Requirements kopieren und installieren (inkl. mlflow, scipy, joblib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt scipy joblib mlflow

# ────────────────────────────────────────────────────────────────────
# 2. Final‑Stage: Nur Code, Modell und bereits installierte Pakete
FROM python:3.10-slim

WORKDIR /app

# 0) Sicherstellen, dass /usr/local/bin im PATH ist
ENV PATH="/usr/local/bin:${PATH}"

# 2a) Kopiere alle Libraries
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10

# 2b) Kopiere alle Console‑Scripts / bin‑Einträge (z.B. gunicorn)
COPY --from=builder /usr/local/bin /usr/local/bin

# 2c) Kopiere deinen Service‑Code, Modell und MLOps‑Skripte
COPY app.py \
     models/rf_model.pkl \
     retrain_model.py \
     drift_detector.py \
     drift_config.json \
     ./

# Port freigeben
EXPOSE 5000

# Produktion: Gunicorn als WSGI‑Server aufrufen
CMD ["python", "-m", "gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]
