"""Integration tests for the DON prediction pipeline."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

# Ensure importing from parent directory works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Register custom layer before importing model
from src.models.attention import MultiHeadSelfAttention
from src.models.don_predictor import DONPredictor
from src.preprocessing.data_processor import DataProcessor
from src.utils.metrics import calculate_metrics


class TestPipeline:
    """Test the complete prediction pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 448
        X = np.random.rand(n_samples, n_features).astype(np.float32)
        y = np.random.rand(n_samples).astype(np.float32)
        
        # Create DataFrame format expected by the processor
        df = pd.DataFrame(X)
        df['vomitoxin_ppb'] = y  # Add the target column with default name
        
        return df
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname
    
    def test_data_processor(self, sample_data, temp_dir):
        """Test data preprocessing pipeline."""
        # Initialize data processor
        processor = DataProcessor()
        
        # Test preprocessing
        X_scaled, y_scaled = processor.preprocess(sample_data)
        
        # Verify the shapes and that scalers are fitted
        assert X_scaled.shape == (len(sample_data), sample_data.shape[1] - 1)
        assert y_scaled.shape == (len(sample_data),)
        assert processor.X_scaler is not None
        assert processor.y_scaler is not None
        
        # Test scaler saving and loading
        x_scaler_path = os.path.join(temp_dir, "x_scaler.pkl")
        y_scaler_path = os.path.join(temp_dir, "y_scaler.pkl")
        processor.save_scalers(x_scaler_path, y_scaler_path)
        
        # Verify scaler files exist
        assert os.path.exists(x_scaler_path)
        assert os.path.exists(y_scaler_path)
        
        # Test loading scalers
        loaded_processor = DataProcessor.load_scalers(x_scaler_path, y_scaler_path)
        
        # Test with the loaded processor
        X_new = np.random.rand(10, X_scaled.shape[1]).astype(np.float32)
        X_scaled_loaded = loaded_processor.scale_features(X_new)
        assert X_scaled_loaded.shape == X_new.shape
        
        # Test inverse transform
        y_pred = np.random.rand(10).astype(np.float32)
        y_pred_original = processor.inverse_transform_target(y_pred)
        assert y_pred_original.shape == y_pred.shape
    
    def test_model_build_and_compile(self, sample_data):
        """Test model building and compilation."""
        # Preprocess data
        processor = DataProcessor()
        X_scaled, y_scaled = processor.preprocess(sample_data)
        
        # Initialize and build model
        model = DONPredictor(input_shape=X_scaled.shape[1])
        built_model = model.build()
        
        # Check the model properties
        assert built_model is not None
        assert isinstance(built_model, tf.keras.Model)
        assert built_model.input_shape[1:] == (X_scaled.shape[1],)
        assert built_model.output_shape[1:] == (1,)
    
    def test_model_save_load(self, sample_data, temp_dir):
        """Test model saving and loading."""
        # Preprocess data
        processor = DataProcessor()
        X_scaled, y_scaled = processor.preprocess(sample_data)
        
        # Initialize and build model
        model = DONPredictor(input_shape=X_scaled.shape[1])
        model.build()
        
        # Save model
        model_path = os.path.join(temp_dir, "test_model.keras")
        model.save(model_path)
        
        # Verify model file exists
        assert os.path.exists(model_path)
        
        # Define custom objects for model loading
        custom_objects = {
            'MultiHeadSelfAttention': MultiHeadSelfAttention
        }
        
        try:
            # Test direct TensorFlow loading (bypassing DONPredictor.load)
            tf_model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            assert tf_model is not None
            
            # Now test with DONPredictor.load
            loaded_model = DONPredictor.load(model_path)
            assert loaded_model is not None
            assert loaded_model.model is not None
            
            # Make a prediction with loaded model to ensure it works
            test_input = np.random.rand(5, X_scaled.shape[1]).astype(np.float32)
            predictions = loaded_model.model.predict(test_input)
            assert predictions.shape == (5, 1)
            
        except Exception as e:
            pytest.fail(f"Model loading failed: {str(e)}")
    
    def test_end_to_end_pipeline(self, sample_data, temp_dir):
        """Test the complete pipeline from data processing to prediction."""
        # Preprocess data
        processor = DataProcessor()
        X_scaled, y_scaled = processor.preprocess(sample_data)
        
        # Split into train and test
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Initialize and build model
        model = DONPredictor(input_shape=X_train.shape[1])
        model.build()
        
        # Train for just 2 epochs (to save time)
        history = model.model.fit(
            X_train, y_train, 
            epochs=2,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training happened
        assert 'loss' in history.history
        assert len(history.history['loss']) == 2
        
        # Make predictions
        y_pred = model.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_test_original = processor.inverse_transform_target(y_test)
        y_pred_original = processor.inverse_transform_target(y_pred.flatten())
        
        # Calculate metrics
        metrics = calculate_metrics(y_test_original, y_pred_original)
        
        # Verify metrics exist
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        
        # Save model and scalers
        model_path = os.path.join(temp_dir, "final_model.keras")
        x_scaler_path = os.path.join(temp_dir, "x_scaler.pkl")
        y_scaler_path = os.path.join(temp_dir, "y_scaler.pkl")
        
        model.save(model_path)
        processor.save_scalers(x_scaler_path, y_scaler_path)
        
        # Verify files exist
        assert os.path.exists(model_path)
        assert os.path.exists(x_scaler_path)
        assert os.path.exists(y_scaler_path)
    
    @pytest.mark.skip(reason="Requires httpx package")
    def test_api_integration(self):
        """Test the FastAPI endpoints."""
        try:
            from fastapi.testclient import TestClient
            from src.api.main import app
            
            client = TestClient(app)
            
            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200
            assert "DON Concentration Prediction API" in response.json()["message"]
            
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            assert "status" in response.json()
            
            # Test prediction endpoint with sample data
            sample = np.random.rand(448).tolist()
            response = client.post("/predict", json={"values": sample})
            
            # Check response structure
            if response.status_code == 200:
                assert "don_concentration" in response.json()
                assert "units" in response.json()
                
        except ImportError:
            pytest.skip("Required packages for API testing not installed")
    
    @pytest.mark.skip(reason="Requires Streamlit context")
    def test_streamlit_components(self):
        """Test Streamlit app components."""
        try:
            import streamlit as st
            from src.streamlit_app import load_model_and_processor
            
            # Test model loading
            model, processor = load_model_and_processor()
            assert model is not None
            assert processor is not None
            
        except ImportError:
            pytest.skip("Streamlit not installed")

if __name__ == "__main__":
    # Run tests with pytest when file is executed directly
    pytest.main([__file__, "-v"])