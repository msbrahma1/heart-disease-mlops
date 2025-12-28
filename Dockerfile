# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask gunicorn

# Copy project files
COPY src/ ./src/
COPY reproduce_model.py .
COPY heart_disease_best_model.pkl .
COPY model_metadata.json .

# Create Flask app file
RUN cat > app.py << 'EOF'
from flask import Flask, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load model at startup
with open('heart_disease_best_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Heart Disease Classifier',
        'version': '1.0'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk"""
    try:
        data = request.get_json()
        
        # Validate input
        required_features = metadata['features']
        if not all(f in data for f in required_features):
            return jsonify({
                'error': 'Missing required features',
                'required': required_features
            }), 400
        
        # Prepare data
        model = model_dict['model']
        scaler = model_dict['scaler']
        imputer = model_dict['imputer']
        feature_names = model_dict['feature_names']
        
        X = np.array([data[f] for f in feature_names]).reshape(1, -1)
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Disease Present' if prediction == 1 else 'No Disease',
            'risk_probability': float(probability[1]),
            'no_disease_probability': float(probability[0]),
            'confidence': float(max(probability))
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'model_name': metadata['model_name'],
        'model_type': metadata['model_type'],
        'features': metadata['features'],
        'metrics': metadata['metrics']
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
