"""Model training module for DON concentration prediction."""

import gc
import logging
import os

import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

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
        self.pbar = tqdm(total=self.total_epochs, desc="Training Progress")

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
            "batch_size": 32,  # Same size as reference
            "epochs": 100,  # Double the epochs to give more time to converge
            "validation_split": 0.20,  # Reduced validation split like reference
            "early_stopping_patience": 10,  # Increased patience for better convergence
            "reduce_lr_patience": 10,  # Increased patience for LR reduction
            "min_lr": 0.00001,  # Keep minimum learning rate
            "model_checkpoint_path": "models/best_model.keras",
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        val_size = int(len(X) * self.config["validation_split"])
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")

        # Convert to tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        # Create datasets with improved shuffling
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), reshuffle_each_iteration=True)
            .batch(self.config["batch_size"])
            .prefetch(tf.data.AUTOTUNE)
            .cache()
        )

        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(self.config["batch_size"])
            .prefetch(tf.data.AUTOTUNE)
            .cache()
        )

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
            # Clear memory before training
            gc.collect()

            # Prepare datasets
            train_dataset, val_dataset = self.prepare_data(X, y)

            # Setup callbacks
            callbacks = [
                ProgressCallback(self.config["epochs"]),
                MemoryUsageCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config["early_stopping_patience"],
                    restore_best_weights=True,
                    min_delta=0.0005,  # More sensitive min delta
                    mode="min",
                    verbose=1,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    self.config["model_checkpoint_path"],
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                    mode="min",
                ),
                # Add TensorBoard for better visualization
                tf.keras.callbacks.TensorBoard(
                    log_dir="./logs", histogram_freq=1, write_graph=True, update_freq="epoch"
                ),
            ]

            # Train model
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config["epochs"],
                callbacks=callbacks,
                verbose=1,  # Show progress for better monitoring
            )

            logger.info("Training completed successfully")

            # Log final metrics
            final_epoch = len(history.history["loss"])
            logger.info(f"Trained for {final_epoch} epochs")
            logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
            logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
            logger.info(f"Final MAE: {history.history['mae'][-1]:.4f}")

            return history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
