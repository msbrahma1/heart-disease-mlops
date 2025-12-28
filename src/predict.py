"""
Prediction script for heart disease classification
"""
import pickle
import numpy as np

def load_model(filepath='heart_disease_best_model.pkl'):
    """Load trained model"""
    with open(filepath, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

def predict(model_dict, features):
    """Make prediction for a patient"""
    model = model_dict['model']
    scaler = model_dict['scaler']
    imputer = model_dict['imputer']
    feature_names = model_dict['feature_names']
    
    X = np.array([features[f] for f in feature_names]).reshape(1, -1)
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'risk_probability': float(probability[1]),
        'confidence': float(max(probability))
    }
