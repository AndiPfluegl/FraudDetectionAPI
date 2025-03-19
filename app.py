import os
import logging
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lade den Modellpfad aus einer Umgebungsvariable, Standard: 'rf_model.pkl'
MODEL_PATH = os.environ.get('MODEL_PATH', 'rf_model.pkl')

# Versuche, das Modell zu laden; falls das fehlschlägt, wird die App es melden
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info("Modell erfolgreich geladen: %s", MODEL_PATH)
except Exception as e:
    logger.error("Fehler beim Laden des Modells: %s", str(e))
    model = None

# Definiere die erwarteten Features (sicherstellen, dass dies mit deinem Training übereinstimmt)
FEATURES = ['V' + str(i) for i in range(1, 29)] + ['Amount_scaled']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modell nicht geladen"}), 500

    try:
        # Erwarte einen JSON-Body mit dem Schlüssel "data"
        content = request.get_json(force=True)
        data = content.get('data', None)
        if data is None:
            return jsonify({"error": "Kein 'data'-Feld im JSON gefunden"}), 400

        # Umwandeln in ein NumPy-Array
        input_data = np.array(data)

        # Überprüfe, ob das Array die richtige Dimension hat
        if input_data.ndim != 2 or input_data.shape[1] != len(FEATURES):
            return jsonify({
                "error": f"Erwarte ein 2D-Array mit {len(FEATURES)} Features, aber erhalten {input_data.shape}"
            }), 400

        # Berechne die Wahrscheinlichkeit für Betrug (Klasse 1)
        fraud_probabilities = model.predict_proba(input_data)[:, 1]
        return jsonify({"fraud_probability": fraud_probabilities.tolist()})
    except Exception as e:
        logger.error("Fehler in /predict: %s", str(e))
        return jsonify({"error": "Interner Serverfehler"}), 500

if __name__ == '__main__':
    # In Produktion sollte debug=False sein und ein WSGI-Server (z.B. Gunicorn) verwendet werden.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
