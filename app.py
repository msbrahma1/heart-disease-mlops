from flask import Flask, request, jsonify
import pickle
import numpy as np
import json
import logging

# Monitoring
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Prometheus metrics exporter
metrics = PrometheusMetrics(app)
metrics.info("app_info", "Heart Disease Prediction API", version="1.0.0")

# Custom metric: count prediction requests
prediction_counter = Counter("prediction_requests_total", "Total prediction requests")

# Load model artifacts
with open("heart_disease_best_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Request received: {data}")
        prediction_counter.inc()  # increment metric

        required_features = metadata["features"]
        if not all(f in data for f in required_features):
            return jsonify({
                "error": "Missing required features",
                "required": required_features
            }), 400

        model = model_dict["model"]
        scaler = model_dict["scaler"]
        imputer = model_dict["imputer"]
        feature_names = model_dict["feature_names"]

        X = np.array([data[f] for f in feature_names]).reshape(1, -1)
        X = imputer.transform(X)
        X = scaler.transform(X)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        response = {
            "prediction": int(prediction),
            "label": "Disease Present" if prediction == 1 else "No Disease",
            "risk_probability": float(probability[1]),
            "confidence": float(max(probability))
        }

        logging.info(f"Prediction response: {response}")
        return jsonify(response), 200

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/info", methods=["GET"])
def info():
    return jsonify(metadata), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)