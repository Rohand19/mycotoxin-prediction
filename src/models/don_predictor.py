"""DON Concentration Predictor Model.

This module implements the deep learning model for predicting DON concentration
in corn samples using hyperspectral data.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import numpy as np

# Use a try-except block to handle different import scenarios
try:
    # When running from project root (for tests)
    from src.models.attention import MultiHeadSelfAttention
except ImportError:
    # When running from within src directory
    from models.attention import MultiHeadSelfAttention

class DONPredictor:
    """Deep learning model for DON concentration prediction."""
    
    def __init__(self, input_shape, config=None):
        """Initialize the DON predictor model.
        
        Args:
            input_shape (int): Number of input features
            config (dict, optional): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config or self._default_config()
        self.model = None
        
    def _default_config(self):
        """Default model configuration."""
        return {
            'l2_lambda': 0.001,
            'attention_heads': 2,
            'attention_dim': 16,
            'dropout_rate': 0.1,
            'dense_layers': [128, 64, 32],
            'learning_rate': 0.001
        }
        
    def build(self):
        """Build the model architecture."""
        l2_lambda = self.config['l2_lambda']
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.input_shape,), name='input'))
        model.add(BatchNormalization(name='batch_norm_1'))
        
        # Dimensionality reduction
        model.add(Dense(256, activation='relu', 
                       kernel_regularizer=l2(l2_lambda), 
                       name='reduction'))
        model.add(BatchNormalization())
        
        # Reshape for attention
        model.add(tf.keras.layers.Reshape((1, 256)))
        
        # Attention mechanism
        model.add(MultiHeadSelfAttention(
            num_heads=self.config['attention_heads'],
            head_dim=self.config['attention_dim'],
            dropout=self.config['dropout_rate'],
            name='attention_1'
        ))
        model.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        
        # Dense layers
        for i, units in enumerate(self.config['dense_layers']):
            model.add(Dense(units, activation='relu',
                          kernel_regularizer=l2(l2_lambda),
                          name=f'dense_{i + 1}'))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate'] * (1 - i * 0.25)))
        
        # Output layer
        model.add(Dense(1, name='output'))
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()
    
    def save(self, filepath):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model must be built before saving")
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model."""
        model = tf.keras.models.load_model(filepath)
        instance = cls(model.input_shape[1])
        instance.model = model
        return instance 