import pytest
import numpy as np
import tensorflow as tf
from model import build_model
from attention import MultiHeadSelfAttention

def test_model_build():
    """Test model building with correct input shape."""
    input_shape = 448
    model = build_model(input_shape)
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 448)
    assert model.output_shape == (None, 1)

def test_attention_mechanism():
    """Test attention mechanism output shape."""
    batch_size = 32
    seq_len = 1
    input_dim = 448
    
    attention = MultiHeadSelfAttention(num_heads=8, head_dim=32)
    inputs = tf.random.normal((batch_size, seq_len, input_dim))
    outputs = attention(inputs)
    
    assert outputs.shape == (batch_size, seq_len, input_dim)

def test_model_prediction():
    """Test model prediction shape and range."""
    input_shape = 448
    model = build_model(input_shape)
    
    # Create dummy input
    X = np.random.normal(size=(10, input_shape))
    
    # Make predictions
    predictions = model.predict(X)
    
    assert predictions.shape == (10, 1)
    assert not np.any(np.isnan(predictions))

def test_model_training():
    """Test model training for one epoch."""
    input_shape = 448
    model = build_model(input_shape)
    
    # Create dummy data
    X = np.random.normal(size=(100, input_shape))
    y = np.random.normal(size=(100, 1))
    
    # Train for one epoch
    history = model.fit(X, y, epochs=1, verbose=0)
    
    assert 'loss' in history.history
    assert len(history.history['loss']) == 1
    assert not np.isnan(history.history['loss'][0])

def test_attention_weights():
    """Test attention weights computation."""
    attention = MultiHeadSelfAttention(num_heads=4, head_dim=32)
    inputs = tf.random.normal((16, 1, 448))
    
    # Get outputs with attention weights
    outputs, attention_weights = attention(inputs, return_attention=True)
    
    assert attention_weights.shape[1] == 4  # num_heads
    assert attention_weights.shape[-2:] == (1, 1)  # attention matrix shape
    assert tf.reduce_sum(tf.abs(tf.reduce_sum(attention_weights, axis=-1) - 1.0)) < 1e-6 