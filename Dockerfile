# 1. Build‑Stage: install Dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# System‑Dependencies (Pandas/NumPy, scipy)
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# copy and install requirements (inkl. mlflow, scipy, joblib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt scipy joblib mlflow

# ────────────────────────────────────────────────────────────────────
# 2. Final‑Stage: code, model and installed packages
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-0 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/bin:${PATH}"

# 2a) copy libraries
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10

# 2b) copy Console‑Scripts / bin‑entries (fe gunicorn)
COPY --from=builder /usr/local/bin /usr/local/bin

# 2c) copy Service‑Code, model und MLOps‑scripts
COPY app.py \
     retrain_model.py \
     drift_detector.py \
     drift_config.json \
     ./

COPY models ./models

# Port:
EXPOSE 5000

# Production: Gunicorn as WSGI‑Server
CMD ["python", "-m", "gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]