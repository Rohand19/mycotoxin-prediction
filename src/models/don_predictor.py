"""DON Concentration Predictor Model.

This module implements the deep learning model for predicting DON concentration
in corn samples using hyperspectral data.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

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
            'attention_heads': 8,
            'attention_dim': 32,
            'dropout_rate': 0.15,
            'dense_layers': [256, 128, 64],
            'learning_rate': 0.0003
        }

    def build(self):
        """Build the model architecture."""
        l2_lambda = self.config["l2_lambda"]
        
        # Create a Sequential model similar to the reference
        model = Sequential([
            # Input layer with explicit dtype
            Input(shape=(self.input_shape,), dtype=tf.float32, name="input"),
            BatchNormalization(name="batch_norm_1"),
            
            # Reshape for attention - critical step
            tf.keras.layers.Reshape((1, self.input_shape)),
            
            # First attention block - 8 heads like reference
            MultiHeadSelfAttention(
                num_heads=self.config["attention_heads"],
                head_dim=self.config["attention_dim"],
                dropout=self.config["dropout_rate"],
                name="attention_1"
            ),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            
            # Second attention block - 4 heads like reference
            MultiHeadSelfAttention(
                num_heads=4,
                head_dim=self.config["attention_dim"],
                dropout=self.config["dropout_rate"],
                name="attention_2"
            ),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            
            # Global average pooling to reduce dimension
            tf.keras.layers.GlobalAveragePooling1D(),
            
            # Dense layers with regularization as in reference
            Dense(256, activation="relu", kernel_regularizer=l2(l2_lambda), name="dense_1"),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(128, activation="relu", kernel_regularizer=l2(l2_lambda), name="dense_2"),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(64, activation="relu", kernel_regularizer=l2(l2_lambda), name="dense_3"),
            BatchNormalization(),
            Dropout(0.15),
            
            # Output layer
            Dense(1, dtype=tf.float32, name="output")
        ])
        
        # Learning rate schedule like reference
        initial_learning_rate = self.config["learning_rate"]
        decay_steps = 3000  # More aggressive decay
        decay_rate = 0.95   # Slightly more aggressive decay rate
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, 
            decay_steps, 
            decay_rate, 
            staircase=True
        )
        
        # Log the learning rate schedule
        print(f"Initial learning rate: {initial_learning_rate}")
        print(f"Learning rate after 10 epochs (~300 steps): {learning_rate_schedule(3000).numpy():.6f}")
        print(f"Learning rate after 50 epochs (~1500 steps): {learning_rate_schedule(15000).numpy():.6f}")
        print(f"Learning rate after 100 epochs (~3000 steps): {learning_rate_schedule(30000).numpy():.6f}")
        
        # Optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_schedule,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Compile with MSE like reference
        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[
                "mae", 
                tf.keras.metrics.RootMeanSquaredError()
            ]
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
        try:
            import tensorflow as tf

            # Disable GPU    
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"Memory growth setting error: {e}")

            try:
                # Try to load the model directly
                print(f"Loading model from {filepath}...")
                model = tf.keras.models.load_model(filepath, compile=False)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise ValueError(f"Failed to load model from {filepath}: {str(e)}")

            # Compile the model after loading
            model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

            instance = cls(model.input_shape[1])
            instance.model = model
            return instance
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    
