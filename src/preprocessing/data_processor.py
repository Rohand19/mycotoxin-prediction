"""Data preprocessing module for DON concentration prediction.

This module handles data loading, cleaning, and preprocessing for the DON
concentration prediction model.
"""

import gc
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processor for DON concentration prediction."""

    def __init__(self, config=None):
        """Initialize the data processor.

        Args:
            config (dict, optional): Configuration parameters for preprocessing
        """
        self.config = config or self._default_config()
        self.X_scaler = None
        self.y_scaler = None
        self._setup_logging()

    def _default_config(self):
        """Default configuration for preprocessing."""
        return {
            "robust_quantile_range": (10.0, 90.0),
            "skewness_threshold": 0.5,
            "target_column": "vomitoxin_ppb",
            "id_column": "hsi_id",
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def load_data(self, file_path):
        """Load data from CSV file.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path, float_precision="high")
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess(self, df):
        """Preprocess the data.

        Args:
            df (pd.DataFrame): Input data

        Returns:
            tuple: Preprocessed features and target (X_scaled, y_scaled)
        """
        logger.info("Starting preprocessing...")
        try:
            # Handle non-numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

            logger.info(f"Found {len(numeric_cols)} numeric and {len(non_numeric_cols)} non-numeric columns")

            if len(non_numeric_cols) > 0:
                logger.info(f"Non-numeric columns: {non_numeric_cols.tolist()}")
                df = df.select_dtypes(include=[np.number])

            # Extract features and target
            y = df[self.config["target_column"]].values
            X = df.drop(columns=[self.config["target_column"]]).values

            # Convert to float32
            X = np.array(X, dtype=np.float32, copy=True)
            y = np.array(y, dtype=np.float32, copy=True)

            # Log initial target statistics
            logger.info(
                f"Target min: {np.min(y):.2f}, max: {np.max(y):.2f}, mean: {np.mean(y):.2f}, median: {np.median(y):.2f}"
            )

            # Clear memory
            del df
            gc.collect()

            # Scale features
            logger.info("Scaling features...")
            self.X_scaler = RobustScaler(quantile_range=self.config["robust_quantile_range"])
            X_scaled = self.X_scaler.fit_transform(X)

            # Analyze target distribution
            logger.info("Analyzing target distribution...")
            skewness = np.abs(np.mean(y) - np.median(y)) / np.std(y)
            logger.info(f"Target skewness: {skewness:.4f}")

            # Apply log transform if needed
            if skewness > self.config["skewness_threshold"]:
                logger.info("Applying log transform to target")
                # Ensure all values are positive before log transform
                min_val = np.min(y)
                if min_val <= 0:
                    logger.info(f"Adjusting target values by {abs(min_val) + 1} to ensure positivity")
                    y = y - min_val + 1  # Add offset to make all values positive
                y = np.log1p(y)  # log(1+x) transform for better handling of small values
                logger.info(f"After log transform - min: {np.min(y):.2f}, max: {np.max(y):.2f}, mean: {np.mean(y):.2f}")

            self.y_scaler = RobustScaler(quantile_range=self.config["robust_quantile_range"])
            y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

            logger.info(
                f"After scaling - y min: {np.min(y_scaled):.2f}, max: {np.max(y_scaled):.2f}, mean: {np.mean(y_scaled):.2f}"
            )
            logger.info("Preprocessing completed")
            logger.info(f"Final shapes - X: {X_scaled.shape}, y: {y_scaled.shape}")

            return X_scaled, y_scaled

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def inverse_transform_target(self, y_scaled):
        """Convert scaled target values back to original scale.

        Args:
            y_scaled (array-like): Scaled target values

        Returns:
            array-like: Original scale target values
        """
        if self.y_scaler is None:
            raise ValueError("Y-scaler not fitted. Run preprocess first.")
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def scale_features(self, X):
        """Scale features using the fitted X_scaler.

        Args:
            X (array-like): Features to scale

        Returns:
            array-like: Scaled features
        """
        if self.X_scaler is None:
            raise ValueError("X-scaler not fitted. Load scalers first.")
        return self.X_scaler.transform(X)

    def save_scalers(self, x_scaler_path, y_scaler_path):
        """Save fitted scalers to disk.

        Args:
            x_scaler_path (str): Path to save X scaler
            y_scaler_path (str): Path to save y scaler
        """
        import joblib

        joblib.dump(self.X_scaler, x_scaler_path)
        joblib.dump(self.y_scaler, y_scaler_path)
        logger.info("Scalers saved successfully")

    @classmethod
    def load_scalers(cls, x_scaler_path, y_scaler_path):
        """Load saved scalers from disk.

        Args:
            x_scaler_path (str): Path to X scaler
            y_scaler_path (str): Path to y scaler

        Returns:
            DataProcessor: Instance with loaded scalers
        """
        import joblib

        instance = cls()
        instance.X_scaler = joblib.load(x_scaler_path)
        instance.y_scaler = joblib.load(y_scaler_path)
        return instance
