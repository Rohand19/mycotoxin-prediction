"""Configuration module for DON concentration prediction."""

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for managing model and training parameters."""

    def __init__(self, config_path=None):
        """Initialize configuration.

        Args:
            config_path (str, optional): Path to YAML config file
        """
        self.config_path = config_path
        self.config = self._load_config() if config_path else self._default_config()

    def _default_config(self):
        """Create default configuration."""
        return {
            "data": {
                "data_path": "data/corn_hyperspectral.csv",
                "target_column": "DON_concentration",
                "id_column": "sample_id",
                "test_size": 0.2,
                "random_state": 42,
                "robust_quantile_range": (25.0, 75.0),
                "skewness_threshold": 0.5,
            },
            "model": {
                "attention_heads": 2,
                "head_dim": 16,
                "dropout_rate": 0.2,
                "l2_lambda": 1e-4,
                "dense_layers": [128, 64, 32],
                "activation": "relu",
                "final_activation": "linear",
            },
            "training": {
                "batch_size": 16,
                "epochs": 100,
                "validation_split": 0.2,
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
                "min_lr": 1e-6,
                "initial_lr": 0.001,
            },
            "paths": {
                "model_dir": "models",
                "logs_dir": "logs",
                "plots_dir": "plots",
                "best_model_path": "models/best_model.keras",
                "scalers_path": "models/scalers.pkl",
            },
        }

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {str(e)}")
            logger.info("Using default configuration")
            return self._default_config()

    def save_config(self, save_path=None):
        """Save configuration to YAML file.

        Args:
            save_path (str, optional): Path to save config file
        """
        try:
            save_path = save_path or self.config_path or "config/config.yaml"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {str(e)}")
            raise

    def update(self, updates):
        """Update configuration with new values.

        Args:
            updates (dict): Dictionary of updates to apply
        """
        try:

            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            self.config = deep_update(self.config, updates)
            logger.info("Configuration updated successfully")

        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise

    def get(self, key, default=None):
        """Get configuration value by key.

        Args:
            key (str): Configuration key (can be nested using dots)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            keys = key.split(".")
            value = self.config
            for k in keys:
                value = value.get(k, default)
                if value == default:
                    break
            return value

        except Exception as e:
            logger.error(f"Error getting configuration value for {key}: {str(e)}")
            return default

    def create_directories(self):
        """Create necessary directories from configuration."""
        try:
            for key, path in self.config["paths"].items():
                if isinstance(path, str) and not path.endswith(".keras") and not path.endswith(".pkl"):
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"Created directory: {path}")

        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise

    def __str__(self):
        """String representation of configuration."""
        return yaml.dump(self.config, default_flow_style=False)
