import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """Class for creating visualizations of spectral data and model results."""
    
    def __init__(self, save_dir: str = "visualizations"):
        """Initialize the visualizer with a directory to save plots.
        
        Args:
            save_dir: Directory where plots will be saved
        """
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_spectral_data(self, data: pd.DataFrame, wavelengths: list, 
                          title: str = "Spectral Reflectance") -> None:
        """Plot spectral reflectance data.
        
        Args:
            data: DataFrame containing spectral data
            wavelengths: List of wavelength values
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        mean_spectrum = data.mean()
        std_spectrum = data.std()
        
        plt.plot(wavelengths, mean_spectrum, 'b-', label='Mean Reflectance')
        plt.fill_between(wavelengths, 
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.2, color='b', label='±1 STD')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/spectral_reflectance.png")
        plt.close()
        
    def plot_data_distribution(self, data: pd.DataFrame, target: np.ndarray,
                             n_features: int = 5) -> None:
        """Plot distribution of features and target variable.
        
        Args:
            data: DataFrame containing feature data
            target: Array of target values
            n_features: Number of features to plot
        """
        # Feature distributions
        plt.figure(figsize=(15, 5))
        for i in range(min(n_features, data.shape[1])):
            plt.subplot(1, n_features, i+1)
            sns.histplot(data.iloc[:, i], kde=True)
            plt.title(f'Feature {i+1}')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_distributions.png")
        plt.close()
        
        # Target distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(target, kde=True)
        plt.title('DON Concentration Distribution')
        plt.xlabel('DON Concentration (ppb)')
        plt.ylabel('Count')
        plt.savefig(f"{self.save_dir}/target_distribution.png")
        plt.close()
        
    def plot_correlation_heatmap(self, data: pd.DataFrame, 
                               n_features: int = 20) -> None:
        """Plot correlation heatmap for features.
        
        Args:
            data: DataFrame containing feature data
            n_features: Number of features to include in heatmap
        """
        plt.figure(figsize=(12, 10))
        selected_cols = data.columns[:n_features]
        sns.heatmap(data[selected_cols].corr(), cmap='coolwarm', center=0,
                   annot=True, fmt='.2f', square=True)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/correlation_heatmap.png")
        plt.close()
        
    def plot_prediction_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                              metrics: Dict[str, float]) -> None:
        """Plot actual vs predicted values and residuals.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            metrics: Dictionary containing evaluation metrics
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        ax1.set_xlabel('Actual DON (ppb)')
        ax1.set_ylabel('Predicted DON (ppb)')
        ax1.set_title('Actual vs Predicted DON Concentration')
        
        # Add metrics annotation
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        ax1.text(0.05, 0.95, metrics_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted DON (ppb)')
        ax2.set_ylabel('Residuals (ppb)')
        ax2.set_title('Residual Analysis')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/prediction_results.png")
        plt.close()
        
    def plot_training_history(self, history: Dict[str, list]) -> None:
        """Plot training history metrics.
        
        Args:
            history: Dictionary containing training history
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_history.png")
        plt.close()
        
    def plot_data_quality(self, data: pd.DataFrame) -> None:
        """Plot data quality metrics and checks.
        
        Args:
            data: DataFrame containing spectral data
        """
        plt.figure(figsize=(15, 5))
        
        # Missing values
        plt.subplot(1, 3, 1)
        missing = data.isnull().sum()
        plt.bar(range(len(missing)), missing)
        plt.title('Missing Values per Feature')
        plt.xlabel('Feature Index')
        plt.ylabel('Count')
        
        # Feature variance
        plt.subplot(1, 3, 2)
        variance = data.var()
        plt.plot(range(len(variance)), variance)
        plt.title('Feature Variance')
        plt.xlabel('Feature Index')
        plt.ylabel('Variance')
        
        # Outlier detection (Z-score based)
        plt.subplot(1, 3, 3)
        z_scores = ((data - data.mean()) / data.std()).abs().mean()
        plt.plot(range(len(z_scores)), z_scores)
        plt.title('Average Absolute Z-scores')
        plt.xlabel('Feature Index')
        plt.ylabel('Mean |Z-score|')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/data_quality.png")
        plt.close() 