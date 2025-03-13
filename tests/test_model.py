"""Unit tests for the DON concentration prediction model."""

import unittest
import sys
import os
import pandas as pd

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the modules
import numpy as np
import tensorflow as tf
from src.models.don_predictor import DONPredictor
from src.models.attention import MultiHeadSelfAttention
from src.preprocessing.data_processor import DataProcessor
from src.utils.metrics import calculate_metrics

class TestDONPredictor(unittest.TestCase):
    """Test cases for DON Predictor model."""
    
    def setUp(self):
        """Set up test cases."""
        self.input_shape = 448  # Number of spectral bands
        self.model = DONPredictor(input_shape=self.input_shape)
        
    def test_model_build(self):
        """Test model building."""
        model = self.model.build()
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape[1], self.input_shape)
        self.assertEqual(model.output_shape[1], 1)
        
    def test_attention_mechanism(self):
        """Test attention mechanism."""
        attention = MultiHeadSelfAttention(num_heads=2, head_dim=16)
        input_shape = (32, 1, 256)  # batch_size, sequence_length, embedding_dim
        test_input = tf.random.normal(input_shape)
        output = attention(test_input)
        self.assertEqual(output.shape, input_shape)
        
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.2, 4.9])
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertTrue(metrics['R2'] <= 1.0)
        
    def test_data_processor(self):
        """Test data processor."""
        processor = DataProcessor()
        
        # Test scaling
        X = np.random.rand(100, self.input_shape)
        y = np.random.rand(100)
        
        # Create a DataFrame with the required columns
        df = pd.DataFrame(X)
        df['vomitoxin_ppb'] = y  # Add the target column
        
        X_scaled, y_scaled = processor.preprocess(df)
        
        self.assertEqual(X_scaled.shape, X.shape)
        self.assertEqual(y_scaled.shape, y.shape)
        # RobustScaler doesn't guarantee zero mean, so we don't check for it
        
if __name__ == '__main__':
    unittest.main() 