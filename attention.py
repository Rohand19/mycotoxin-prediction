import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import numpy as np

class SelfAttention(Layer):
    def __init__(self, attention_dim=32, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.query = Dense(self.attention_dim, use_bias=False)
        self.key = Dense(self.attention_dim, use_bias=False)
        self.value = Dense(self.attention_dim, use_bias=False)
        self.attention_weights = Dense(1, use_bias=False)
        
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Reshape input to (batch_size, sequence_length, features)
        batch_size = tf.shape(inputs)[0]
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        
        # Compute Q, K, V
        query = self.query(inputs)  # (batch_size, seq_len, attention_dim)
        key = self.key(inputs)      # (batch_size, seq_len, attention_dim)
        value = self.value(inputs)  # (batch_size, seq_len, attention_dim)
        
        # Compute attention scores
        score = tf.matmul(query, key, transpose_b=True)  # (batch_size, seq_len, seq_len)
        score = score / tf.math.sqrt(tf.cast(self.attention_dim, tf.float32))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention weights to values
        context_vector = tf.matmul(attention_weights, value)  # (batch_size, seq_len, attention_dim)
        
        # Compute final attention weights
        attention_output = self.attention_weights(context_vector)  # (batch_size, seq_len, 1)
        attention_output = tf.squeeze(attention_output, axis=-1)  # (batch_size, seq_len)
        
        # If input was 2D, return 2D output
        if len(inputs.shape) == 2:
            attention_output = tf.squeeze(attention_output, axis=1)
            
        return attention_output

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'attention_dim': self.attention_dim
        })
        return config

class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads=4, head_dim=32, dropout=0.1, use_bias=True, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.output_dim = num_heads * head_dim

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Initialize Q, K, V dense layers with bias
        self.query_dense = Dense(self.output_dim, use_bias=self.use_bias)
        self.key_dense = Dense(self.output_dim, use_bias=self.use_bias)
        self.value_dense = Dense(self.output_dim, use_bias=self.use_bias)
        
        # Add layer normalization
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Output projection to match input dimension
        self.combine_heads = Dense(input_dim)
        
        # Dropout layer for attention weights
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        
        super(MultiHeadSelfAttention, self).build(input_shape)

    def attention(self, query, key, value):
        # Scaled dot-product attention
        score = tf.matmul(query, key, transpose_b=True)
        
        # Scale scores
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_score, axis=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout_layer(attention_weights, training=True)
        
        # Compute context vector
        output = tf.matmul(attention_weights, value)
        
        return output, attention_weights

    def separate_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None, return_attention=False):
        batch_size = tf.shape(inputs)[0]
        
        # Apply layer normalization first (pre-norm formulation)
        x = self.layernorm(inputs)
        
        # Linear projections
        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)
        
        # Separate heads
        query = self.separate_heads(query)
        key = self.separate_heads(key)
        value = self.separate_heads(value)
        
        # Apply attention
        attention_output, attention_weights = self.attention(query, key, value)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.output_dim))
        
        # Final linear layer to match input dimension
        output = self.combine_heads(concat_attention)
        
        # Add residual connection
        output = output + inputs
        
        if return_attention:
            return output, attention_weights
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout': self.dropout,
            'use_bias': self.use_bias
        })
        return config 