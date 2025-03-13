"""Model training module for DON concentration prediction."""

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import logging
import os
import gc
import psutil

logger = logging.getLogger(__name__)

class MemoryUsageCallback(Callback):
    """Callback to monitor memory usage during training."""
    
    def on_epoch_begin(self, epoch, logs=None):
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage at epoch start: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    def on_epoch_end(self, epoch, logs=None):
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage at epoch end: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        gc.collect()

class ProgressCallback(Callback):
    """Custom progress bar callback."""
    
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

class ModelTrainer:
    """Trainer for DON concentration prediction model."""
    
    def __init__(self, model, config=None):
        """Initialize the trainer.
        
        Args:
            model: The model to train
            config (dict, optional): Training configuration
        """
        self.model = model
        self.config = config or self._default_config()
        self._setup_logging()
    
    def _default_config(self):
        """Default training configuration."""
        return {
            'batch_size': 16,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'min_lr': 0.00001,
            'model_checkpoint_path': 'models/best_model.keras'
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def prepare_data(self, X, y):
        """Prepare data for training.
        
        Args:
            X (array-like): Input features
            y (array-like): Target values
            
        Returns:
            tuple: Training and validation datasets
        """
        # Create validation split
        val_size = int(len(X) * self.config['validation_split'])
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        
        # Convert to tensors
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)
        ).shuffle(1024).batch(self.config['batch_size'])
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)
        ).batch(self.config['batch_size'])
        
        return train_dataset, val_dataset
    
    def train(self, X, y):
        """Train the model.
        
        Args:
            X (array-like): Input features
            y (array-like): Target values
            
        Returns:
            dict: Training history
        """
        logger.info("Starting training...")
        logger.info(f"Input shapes - X: {X.shape}, y: {y.shape}")
        
        try:
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_data(X, y)
            
            # Setup callbacks
            callbacks = [
                ProgressCallback(self.config['epochs']),
                MemoryUsageCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['early_stopping_patience'],
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config['reduce_lr_patience'],
                    min_lr=self.config['min_lr'],
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    self.config['model_checkpoint_path'],
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config['epochs'],
                callbacks=callbacks,
                verbose=0
            )
            
            logger.info("Training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise 