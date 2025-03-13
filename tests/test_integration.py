"""Integration tests for the DON prediction pipeline."""
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tempfile
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.don_predictor import DONPredictor
from src.preprocessing.data_processor import DataProcessor
from src.utils.interpretability import ModelInterpreter

class TestPipeline:
    """Test the complete prediction pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 448
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        return pd.DataFrame(X), pd.Series(y)
        
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname
            
    def test_end_to_end_pipeline(self, sample_data, temp_model_dir):
        """Test the complete pipeline from data processing to prediction."""
        X, y = sample_data
        
        # Initialize components
        processor = DataProcessor()
        model = DONPredictor()
        
        # Test data processing
        X_scaled = processor.scale_features(X)
        assert X_scaled.shape == X.shape
        
        # Test model training
        model.build((None, X.shape[1]))
        history = model.model.fit(
            X_scaled, y,
            epochs=2,
            validation_split=0.2,
            verbose=0
        )
        assert 'loss' in history.history
        
        # Test model saving and loading
        model_path = os.path.join(temp_model_dir, 'test_model.keras')
        model.save(model_path)
        loaded_model = DONPredictor.load(model_path)
        assert isinstance(loaded_model, DONPredictor)
        
        # Test prediction
        y_pred = loaded_model.model.predict(X_scaled)
        assert y_pred.shape[0] == len(y)
        
        # Test interpretability
        interpreter = ModelInterpreter(loaded_model.model)
        interpreter.prepare_explainer(X_scaled[:10])
        shap_values, summary = interpreter.explain_prediction(X_scaled[:1])
        assert isinstance(summary, dict)
        assert 'mean_impact' in summary
        
    def test_api_integration(self, sample_data):
        """Test the FastAPI endpoints."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # Test prediction endpoint
        X, _ = sample_data
        sample_input = {"values": X.iloc[0].tolist()}
        response = client.post("/predict", json=sample_input)
        assert response.status_code == 200
        assert "don_concentration" in response.json()
        
    def test_streamlit_components(self, sample_data):
        """Test Streamlit app components."""
        import streamlit as st
        from src.streamlit_app import load_model_and_processor
        
        # Test model loading
        model, processor = load_model_and_processor()
        assert model is not None
        assert processor is not None
        
        # Test data processing in Streamlit
        X, _ = sample_data
        sample = X.iloc[0].values.reshape(1, -1)
        X_scaled = processor.scale_features(sample)
        assert X_scaled.shape == sample.shape