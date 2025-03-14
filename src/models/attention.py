"""Attention mechanism for the DON concentration prediction model."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


# Register the custom attention layer for serialization
@tf.keras.utils.register_keras_serializable(package="src.models")
class MultiHeadSelfAttention(Layer):
    """Multi-head self-attention mechanism.

    This layer implements the multi-head self-attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        dropout (float): Dropout rate
        use_bias (bool): Whether to use bias in attention calculations
    """

    def __init__(self, num_heads=2, head_dim=16, dropout=0.1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_bias = use_bias

        self.output_dim = num_heads * head_dim

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        # Query, Key, and Value weight matrices
        self.query_dense = tf.keras.layers.Dense(
            self.output_dim, use_bias=self.use_bias, name="query", dtype=tf.float32
        )
        self.key_dense = tf.keras.layers.Dense(self.output_dim, use_bias=self.use_bias, name="key", dtype=tf.float32)
        self.value_dense = tf.keras.layers.Dense(
            self.output_dim, use_bias=self.use_bias, name="value", dtype=tf.float32
        )

        # Output projection
        self.combine_heads = tf.keras.layers.Dense(
            self.input_dim, use_bias=self.use_bias, name="output", dtype=tf.float32
        )

        # Dropout layer
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def attention(self, query, key, value, mask=None):
        """Compute scaled dot-product attention."""
        # Ensure inputs are float32
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)
        value = tf.cast(value, tf.float32)

        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            scaled_score += mask * -1e9

        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout_layer(weights)

        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x):
        """Separate the input into multiple heads."""
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        # Reshape for multi-head attention
        x = tf.reshape(x, (batch_size, length, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, return_attention=False, training=None):
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        # Linear transformations
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Separate heads
        query = self.separate_heads(query)
        key = self.separate_heads(key)
        value = self.separate_heads(value)

        # Attention
        output, weights = self.attention(query, key, value, mask)

        # Combine heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, length, self.output_dim))
        output = self.combine_heads(output)

        if return_attention:
            return output, weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "dropout": self.dropout,
                "use_bias": self.use_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create a MultiHeadSelfAttention layer from its config.

        Args:
            config: Dictionary with the configuration

        Returns:
            A new MultiHeadSelfAttention instance
        """
        return cls(**config)
