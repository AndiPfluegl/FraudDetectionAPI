Fraud Detection API

This repository provides a simple fraud detection RESTful service, integrated into an automated MLOps pipeline using GitHub Actions and self‑hosted Docker volumes.

Project Structure

├── app.py                     # Flask application exposing /predict and /metrics
├── import_csv_to_sqlite.py    # Utility: build initial reference DB from CSV
├── drift_detector.py          # KS‑test based data drift detector
├── retrain_model.py           # Retrain RandomForest on SQLite data
├── Dockerfile                 # Dockerfile
├── .github/workflows/         # CI/CD pipeline definitions
│   └── mlops.yml
├── data/
│   ├── reference_data.db      # Seed reference database (versioned)
│   └── drift_config.json      # Features & p‑value threshold config
├── models/                    # Trained model artifacts (Pickle)
└── requirements.txt           # Python dependencies

Quick Start

Build and run the API

docker build -t fraud-api:latest .
docker run -d --name fraud_api \
  -v fraud-data:/app/data \
  -e API_TOKEN="<your-token>" \
  -e LATEST_DB=/app/data/requests.db \
  -e FRAUD_THRESHOLD=0.4 \
  -p 5000:5000 fraud-api:latest

Send a test prediction

curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{"data":[[...28 feature values..., 0.99]]}'

Inspect metrics

curl http://localhost:5000/metrics -H "Authorization: Bearer <your-token>"

MLOps Pipeline

The GitHub Actions workflow (.github/workflows/mlops.yml) automates:

deploy_job – Extract latest requests.db from fraud-data volume.

drift_job – Compare reference_data.db ↔ requests.db via KS‑test.

retrain_job – Retrain model if drift detected or on monthly schedule.

update_reference – Overwrite the fraud-reference volume with the latest requests.db.

build-and-push – Build and push updated Docker image with new model.

deploy_final – Restart local container with fraud-api:latest.

Cleaning Up the Repo

Remove raw CSVs and temporary files; only keep data/reference_data.db and drift_config.json in /data.

Add /data/requests.db and /app/data/* to .gitignore to avoid committing runtime data.

Ensure /models only contains versioned model artefacts (if any).

Code Comments

All Python scripts are documented with English comments explaining:

Configuration (environment variables and defaults)

Data loading (CSV vs SQLite)

Drift detection logic

Retraining and model logging

Please review each script and fill in any missing docstrings or inline comments.
