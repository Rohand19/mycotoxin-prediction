"""Main training script for DON concentration prediction model."""

import os
import logging
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from models.don_predictor import DONPredictor
from models.trainer import ModelTrainer
from preprocessing.data_processor import DataProcessor
from utils.visualization import plot_training_history, plot_predictions
from utils.metrics import calculate_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
        
        # Load and preprocess data
        data_processor = DataProcessor()
        df = data_processor.load_data('data/corn_hyperspectral.csv')
        X_scaled, y_scaled = data_processor.preprocess(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled,
            test_size=0.2,
            random_state=42
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        # Initialize and build model
        model = DONPredictor(input_shape=X_train.shape[1])
        model.build()
        model.summary()
        
        # Train model
        trainer = ModelTrainer(model.model)
        history = trainer.train(X_train, y_train)
        
        # Save model and scalers
        model.save('models/final_model.keras')
        data_processor.save_scalers(
            'models/X_scaler.pkl',
            'models/y_scaler.pkl'
        )
        
        # Generate visualizations
        plot_training_history(history, save_path='visualizations/training_history.png')
        
        # Make predictions on test set
        y_pred = model.model.predict(X_test)
        y_test_original = data_processor.inverse_transform_target(y_test)
        y_pred_original = data_processor.inverse_transform_target(y_pred)
        
        # Calculate and log metrics
        metrics = calculate_metrics(y_test_original, y_pred_original)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Plot predictions
        plot_predictions(
            y_test_original,
            y_pred_original,
            save_path='visualizations/predictions.png'
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 