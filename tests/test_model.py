"""Unit tests for model training and predictions"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TestModelTraining:
    """Test model training and evaluation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = pd.DataFrame({
            'age': np.random.randint(30, 80, 100),
            'sex': np.random.randint(0, 2, 100),
            'cp': np.random.randint(1, 5, 100),
            'trestbps': np.random.randint(90, 180, 100),
            'chol': np.random.randint(150, 400, 100),
            'fbs': np.random.randint(0, 2, 100),
            'restecg': np.random.randint(0, 3, 100),
            'thalach': np.random.randint(60, 200, 100),
            'exang': np.random.randint(0, 2, 100),
            'oldpeak': np.random.rand(100) * 6,
            'slope': np.random.randint(1, 4, 100),
            'ca': np.random.rand(100) * 4,
            'thal': np.random.rand(100) * 8,
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y
    
    def test_train_test_split(self, sample_data):
        """Test train-test split"""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        assert len(X_train) == 80, "Train set size incorrect"
        assert len(X_test) == 20, "Test set size incorrect"
    
    def test_model_training(self, sample_data):
        """Test model training"""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'predict'), "Missing predict method"
        assert hasattr(model, 'score'), "Missing score method"
    
    def test_model_prediction(self, sample_data):
        """Test model predictions"""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test), "Prediction count mismatch"
        assert set(predictions).issubset({0, 1}), "Predictions not binary"
    
    def test_model_scoring(self, sample_data):
        """Test model scoring"""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        assert 0 <= score <= 1, "Score out of range"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
