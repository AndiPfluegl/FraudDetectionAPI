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
Create and Activate a Virtual Environment:

bash
Kopieren
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
Install the Dependencies:

bash
Kopieren
pip install -r requirements.txt
Running the Application Locally
Start the Flask Application:

bash
Kopieren
python app.py
Access the API: The API will be available at http://0.0.0.0:5000.

Usage
Endpoint: /predict

Method: POST

Payload Format: The API expects a JSON object with a key "data", which is a list of observations. Each observation should be a list of feature values matching the order used during model training.

Example using cURL:

bash
Kopieren
curl -X POST -H "Content-Type: application/json" -d '{"data": [[0.1, -1.2, 0.5, 0.3, 0.0, 0.2, -0.4, 1.1, -0.3, 0.7, 0.5, -0.1, 0.2, 0.3, -0.2, 0.4, 0.6, -0.5, 0.0, 0.1, 0.2, -0.3, 0.4, 0.5, 0.6, -0.1, 0.0, 0.2, 0.5]]}' http://0.0.0.0:5000/predict
Response: A JSON response containing the fraud probability:

json
Kopieren
{
  "fraud_probability": [0.15]
}
Deployment
This application is designed to be deployed on Azure App Service. For deployment:

Use GitHub integration for automated deployments.
Configure Application Settings in the Azure Portal (e.g., MODEL_PATH and PORT).
Set up monitoring with Application Insights.
For detailed deployment instructions, refer to the Azure documentation and the project’s deployment guide.

License
This project is licensed under the MIT License.

Acknowledgments
This project was developed as part of a university assignment to explore fraud detection using machine learning.

pgsql
Kopieren

Simply copy all the text above and paste it into your `README.md` file. Feel free to modify any se
