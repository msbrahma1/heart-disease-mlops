"""Unit tests for data processing"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class TestDataProcessing:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'age': [63, 67, 37, 41, 56],
            'sex': [1, 1, 1, 0, 1],
            'cp': [1, 4, 3, 2, 2],
            'trestbps': [145, 160, 130, 130, 120],
            'chol': [233, 286, 250, 204, 236],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [2, 2, 0, 2, 0],
            'thalach': [150, 108, 187, 172, 178],
            'exang': [0, 1, 0, 0, 0],
            'oldpeak': [2.3, 1.5, 3.5, 1.4, 0.8],
            'slope': [3, 2, 3, 1, 1],
            'ca': [0.0, 3.0, 0.0, 0.0, 0.0],
            'thal': [6.0, 3.0, 3.0, 3.0, 3.0],
            'num_binary': [0, 1, 0, 0, 0]
        })
    
    def test_data_shape(self, sample_data):
        assert sample_data.shape[0] > 0, "Data is empty"
        assert sample_data.shape[1] >= 13, "Missing columns"
    
    def test_no_missing_values(self, sample_data):
        assert sample_data.isnull().sum().sum() == 0, "Missing values found"
    
    def test_binary_target(self, sample_data):
        assert set(sample_data['num_binary'].unique()).issubset({0, 1}), "Target not binary"
    
    def test_feature_ranges(self, sample_data):
        assert sample_data['age'].min() > 0, "Age out of range"
        assert sample_data['age'].max() < 150, "Age out of range"
    
    def test_imputation(self, sample_data):
        data_with_missing = sample_data.copy()
        data_with_missing.iloc[0, 0] = np.nan
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data_with_missing)
        assert not np.isnan(data_imputed).any(), "Imputation failed"
    
    def test_scaling(self, sample_data):
        X = sample_data.drop('num_binary', axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape, "Scaling shape mismatch"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
