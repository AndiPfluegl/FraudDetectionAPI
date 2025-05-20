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
8. [Architecture Overview](#architecture-overview)  
9. [Configuration](#configuration)  
10. [Contributing](#contributing)  
11. [License](#license)  

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

   
