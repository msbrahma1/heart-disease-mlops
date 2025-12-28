"""Unit tests for data processing"""
import pytest
import pandas as pd
import numpy as np

class TestDataProcessing:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'age': [63, 67, 37],
            'sex': [1, 1, 1],
            'num_binary': [0, 1, 0]
        })
    
    def test_data_shape(self, sample_data):
        assert sample_data.shape[0] > 0
    
    def test_no_missing(self, sample_data):
        assert sample_data.isnull().sum().sum() == 0
