"""
Reproducibility Script: Load and test the saved model
"""
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_model(model_path='heart_disease_best_model.pkl'):
    """Load the saved model with preprocessors"""
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

def predict_single_patient(model_dict, patient_features):
    """
    Predict heart disease risk for a single patient
    
    Args:
        model_dict: Dictionary with model, scaler, imputer, features
        patient_features: Dictionary with patient health data
    
    Returns:
        Dictionary with prediction and probability
    """
    model = model_dict['model']
    scaler = model_dict['scaler']
    imputer = model_dict['imputer']
    feature_names = model_dict['feature_names']
    
    # Create feature array in correct order
    X_patient = np.array([patient_features[f] for f in feature_names]).reshape(1, -1)
    
    # Preprocess (same as training)
    X_imputed = imputer.transform(X_patient)
    X_scaled = scaler.transform(X_imputed)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'risk_probability': float(probability[1]),
        'confidence': float(max(probability)),
        'prediction_label': 'Disease Present' if prediction == 1 else 'No Disease'
    }

if __name__ == "__main__":
    # Example usage
    model_dict = load_model()
    
    # Example patient
    example_patient = {
        'age': 63, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3,
        'slope': 3, 'ca': 0.0, 'thal': 6.0
    }
    
    result = predict_single_patient(model_dict, example_patient)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Risk Probability: {result['risk_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
