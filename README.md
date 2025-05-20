# Fraud Detection API & MLOps Pipeline

This repository contains a complete end-to-end solution for a fraud detection service, including:

- A Flask-based RESTful API that serves a trained Random Forest model to predict fraud probability.  
- Persistent storage of prediction requests in SQLite (via Docker volume).  
- Data drift detection using a Kolmogorov–Smirnov test to detect shifts in feature distributions.  
- Automated model retraining and redeployment using GitHub Actions and a self-hosted runner.  
- Monitoring of service metrics (prediction counts, probability distributions) via Prometheus & Grafana.  
- PushGateway integration to report model accuracy to Prometheus.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Setup & Installation](#setup--installation)  
3. [Running the Fraud Detection API](#running-the-fraud-detection-api)  
4. [API Usage](#api-usage)  
5. [Data Drift Detection & Retraining](#data-drift-detection--retraining)  
6. [CI/CD Pipeline](#cicd-pipeline)  
7. [Monitoring](#monitoring)  
8. [Configuration](#configuration)    

---

## Prerequisites

- Docker & Docker Compose  
- Python 3.10+ (if running scripts locally)  
- GitHub account with permissions to set up Secrets  
- (Optional) Grafana for dashboarding  

---

## Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AndiPfluegl/FraudDetectionAPI.git
   cd FraudDetectionAPI

2. **Configure environment variables (either in .env or shell)**:

   ```bash
   export API_TOKEN="your_secure_token"
   export FRAUD_THRESHOLD=0.4

3. **Create Docker volumes:**:

   ```bash
   docker volume create fraud-data
   docker volume create fraud-reference

4. **Build & run all containers (API, Prometheus, Pushgateway, Grafana)**:

   ```bash
   docker-compose up --build -d

## Running the Fraud Detection API

The Flask API listens on port 5000 and exposes:
- POST /predict: returns fraud probability for given feature vectors.
- GET /metrics: Prometheus-formatted metrics (requires Authorization: Bearer $API_TOKEN).

   ```bash
   docker run -d --name fraud_api \
     -v fraud-data:/app/data \
     --env API_TOKEN=$API_TOKEN \
     --env FRAUD_THRESHOLD=$FRAUD_THRESHOLD \
     -p 5000:5000 \
     ghcr.io/andipfluegl/fraud-api:latest
   ```
---


## API Usage

PowerShell one-liner:
   ```powershell
   Invoke-RestMethod -Method POST -Uri http://localhost:5000/predict `
     -Headers @{ Authorization = 'Bearer h2k9fj3lK_s8Df0WpQxY4Z7bP2nA1sR6'; 'Content-Type' = 'application/json' } `
     -Body '{"data":[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,0.99]]}'
   ```

cURL + Python formatter:

```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer h2k9fj3lK_s8Df0WpQxY4Z7bP2nA1sR6" \
  -H "Content-Type: application/json" \
  -d '{"data":[[0.1,0.2,…,0.99]]}' \
| python -m json.tool
   ```

Check latest saved request:

```bash
docker exec fraud_api sqlite3 /app/data/requests.db "SELECT * FROM requests ORDER BY ts DESC LIMIT 1;"
```
---

## Data Drift Detection & Retraining

- Drift Detector: drift_detector.py uses KS-tests to compare distributions in reference_data.db vs. requests.db.
- Retraining: retrain_model.py runs on drift or monthly schedule:
  1. Reads rows with true Class or derived labels from score.
  2. Performs stratified split.
  3. Trains a RandomForestClassifier.
  4. Logs accuracy to MLflow & Pushgateway.
  5. Exports models/rf_model.pkl.
---

## CI/CD Pipeline

Workflow .github/workflows/mlops.yml:
1. deploy_job: Extracts DBs from Docker volumes.
2. drift_job: Runs drift detection, outputs drift flag.
3. retrain_job: Retrains model if drift or on schedule.
4. update_reference: Overwrites reference_data.db with latest data.
5. build-and-push: Builds & pushes Docker image to GHCR.
6. deploy_final: Restarts the local fraud_api container.

GitHub Secrets:
- API_TOKEN: for API auth
- GITHUB_TOKEN: for GHCR login
---

## Monitoring
- Prometheus scrapes:
  - fraud_api:5000/metrics
  - pushgateway:9091/metrics
- Pushgateway receives model_accuracy from retraining.
- Grafana dashboards for:
  - predictions_total (counter),
  - fraud_probabilities (histogram),
  - model_accuracy (gauge).
---
## Configuration

| Variable          | Default                | Description                                    |
|-------------------|------------------------|------------------------------------------------|
| `API_TOKEN`       | required               | Bearer token for API auth                      |
| `FRAUD_THRESHOLD` | 0.4                    | Threshold for classifying fraud vs legit       |
| `MODEL_PATH`      | `models/rf_model.pkl`  | Path to the pickled model                      |
| `LATEST_DB`       | `data/requests.db`     | Requests DB inside container                   |
| `TRAIN_DB`        | `data/requests.db`     | Used by retrain script                         |
| `MLFLOW_EXPER`    | `FraudDetection`       | MLflow experiment name                         |
| `MLFLOW_REG`      | `fraud-api`            | MLflow registered model name                   |
