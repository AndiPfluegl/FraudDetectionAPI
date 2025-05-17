--- Build stage ---------------------------------------------------------

Install Python dependencies into an isolated layer to speed up rebuilds

FROM python:3.10-slim AS builder

WORKDIR /app

Install system build tools required by pandas, numpy, and other compiled packages

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

Copy and install Python requirements, including MLflow, SciPy, and joblib

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt scipy joblib mlflow

--- Final stage ---------------------------------------------------------

Use a clean Python base image to minimize final image size

FROM python:3.10-slim

WORKDIR /app

Install SQLite CLI and library for DB sanity checks if needed

RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 libsqlite3-0 && rm -rf /var/lib/apt/lists/*

Ensure /usr/local/bin is in PATH for pip-installed console scripts (e.g., gunicorn)

ENV PATH="/usr/local/bin:${PATH}"

Copy installed Python packages from the builder image

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

Copy application code, drift & retrain scripts, config, and model artifacts

COPY app.py retrain_model.py drift_detector.py import_csv_to_sqlite.py drift_config.json ./
COPY models ./models

Expose the HTTP port for the Flask/Gunicorn service

EXPOSE 5000

Use Gunicorn as production WSGI server with 2 workers

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]