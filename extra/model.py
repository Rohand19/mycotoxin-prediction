from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import sys
import gc
import time
from attention import MultiHeadSelfAttention

# Force CPU usage - more aggressive approach
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Clear any existing models/memory
tf.keras.backend.clear_session()
gc.collect()

# Disable GPU
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    print("No GPU devices found to disable")

# Print system information
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Eager execution enabled:", tf.executing_eagerly())
print("Available devices:", tf.config.list_physical_devices())

# Define the TQDMProgressBar callback class
class TQDMProgressBar(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs, desc='Training Progress')

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        self.pbar.set_postfix(logs)

    def on_train_end(self, logs=None):
        self.pbar.close()

# Model building function with enhanced error handling and smaller network
def build_model(input_shape):
    print(f"Building model with input shape: {input_shape}")
    try:
        # Create a model with L2 regularization and attention
        l2_lambda = 0.001  # L2 regularization factor
        
        model = Sequential([
            Input(shape=(input_shape,), name='input'),
            BatchNormalization(name='batch_norm_1'),
            
            # Reshape for attention
            tf.keras.layers.Reshape((1, input_shape)),
            
            # First attention block with matching dimensions
            MultiHeadSelfAttention(num_heads=8, head_dim=32, dropout=0.15, name='attention_1'),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            
            # Second attention block
            MultiHeadSelfAttention(num_heads=4, head_dim=32, dropout=0.15, name='attention_2'),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            
            # Global average pooling to reduce dimension
            tf.keras.layers.GlobalAveragePooling1D(),
            
            # Dense layers with improved regularization
            Dense(256, activation='relu', kernel_regularizer=l2(l2_lambda), name='dense_1'),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda), name='dense_2'),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda), name='dense_3'),
            BatchNormalization(),
            Dropout(0.15),
            
            Dense(1, name='output')
        ])
        
        print("Model created successfully")
        print("\nModel structure:")
        model.summary()
        
        # Learning rate schedule with slower decay
        initial_learning_rate = 0.0003  # Reduced initial learning rate
        decay_steps = 5000
        decay_rate = 0.98
        learning_rate_schedule = ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate, staircase=True
        )
        
        print("\nCompiling model...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_schedule,
            clipnorm=1.0  # Add gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error'
        )
        print("Model compilation complete")
        return model
        
    except Exception as e:
        print(f"Error in build_model: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise

# Updated train_model function with enhanced error handling and smaller batch size
def train_model(model, X_train, y_train, epochs=10, batch_size=32, callbacks=None):
    print(f"\nStarting training with:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Clear memory before training
    gc.collect()
    
    try:
        # Convert data to tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        # Calculate number of batches
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Number of batches per epoch: {n_batches}")
        
        # Training history
        history = {'loss': [], 'val_loss': []}
        
        # Create validation set
        val_size = n_samples // 5  # 20% for validation
        indices = tf.random.shuffle(tf.range(n_samples))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_val = tf.gather(X_train, val_indices)
        y_val = tf.gather(y_train, val_indices)
        X_train_final = tf.gather(X_train, train_indices)
        y_train_final = tf.gather(y_train, train_indices)
        
        best_val_loss = float('inf')
        best_weights = None
        patience = 15  # Increased patience
        patience_counter = 0
        min_delta = 0.0005  # Reduced minimum improvement threshold
        
        # Custom training loop with validation
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            start_time = time.time()
            epoch_loss = 0
            
            # Shuffle training data
            train_indices = tf.random.shuffle(tf.range(len(X_train_final)))
            X_shuffled = tf.gather(X_train_final, train_indices)
            y_shuffled = tf.gather(y_train_final, train_indices)
            
            # Training
            for batch in range((len(X_train_final) + batch_size - 1) // batch_size):
                try:
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, len(X_train_final))
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    
                    with tf.GradientTape() as tape:
                        y_pred = model(X_batch, training=True)
                        batch_loss = tf.keras.losses.mean_squared_error(y_batch, tf.squeeze(y_pred))
                        batch_loss = tf.reduce_mean(batch_loss)
                        
                        # Add L2 regularization loss
                        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                          if 'kernel' in v.name]) * 0.001
                        total_loss = batch_loss + l2_loss
                    
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    
                    epoch_loss += batch_loss.numpy()
                    
                    if (batch + 1) % 5 == 0 or (batch + 1) == n_batches:
                        print(f"Batch {batch + 1}/{n_batches} - Current loss: {batch_loss:.4f}", flush=True)
                    
                except Exception as e:
                    print(f"Error in batch {batch}: {str(e)}")
                    raise
            
            # Validation
            val_predictions = model(X_val, training=False)
            val_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_val, tf.squeeze(val_predictions)))
            
            # Compute metrics
            epoch_loss /= n_batches
            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss.numpy())
            
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Loss: {epoch_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Time: {time.time() - start_time:.2f}s")
            
            # Save best weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Restore best weights
        if best_weights is not None:
            model.set_weights(best_weights)
            print("Restored best weights from validation")
        
        print("\nTraining completed successfully")
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise