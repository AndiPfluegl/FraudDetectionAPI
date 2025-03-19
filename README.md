# Fraud Detection API

This repository contains a Flask-based RESTful API for fraud detection using a pre-trained Random Forest model. It was developed as part of a project to detect fraudulent financial transactions and provide a scalable, cloud-ready solution.

## Project Structure

- **app.py**: The main Flask application defining the `/predict` endpoint.
- **rf_model.pkl**: The pre-trained Random Forest model serialized with Pickle.
- **requirements.txt**: A list of required Python packages.
- **README.md**: This documentation file.

## Features

- **REST API Endpoint**: Accepts POST requests with JSON payloads containing feature data.
- **Fraud Prediction**: Returns the probability of a transaction being fraudulent.
- **Production-Ready Configuration**: Uses environment variables and proper logging for secure deployment.

## Prerequisites

- Python 3.8 or higher
- A virtual environment is recommended
- Key dependencies (see `requirements.txt` for the full list):
  - Flask
  - scikit-learn
  - numpy
  - pandas
  - gunicorn (recommended for production deployment)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/FraudDetectionAPI.git
   cd FraudDetectionAPI
2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
3. **Install the Dependencies:**
   ```bash
   pip install -r requirements.txt

## Running the Application Locally
1. **Start the Flask Application**
   ```bash
   python app.py
2. **Access the API:**
   The API will be available at http://0.0.0.0:5000.

## Usage
- **Endpoint**: /predict
