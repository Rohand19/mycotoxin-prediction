"""Data quality analysis and visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """Class for analyzing and visualizing data quality."""
    
    def __init__(self):
        """Initialize the analyzer."""
        plt.style.use('default')
    
    def plot_spectral_curves(self, data, wavelengths=None, save_path=None):
        """Plot spectral reflectance curves.
        
        Args:
            data (DataFrame): Spectral data
            wavelengths (array-like, optional): Wavelength values
            save_path (str, optional): Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 6))
            
            if wavelengths is None:
                wavelengths = range(data.shape[1])
            
            # Plot mean curve with confidence interval
            mean_spectrum = data.mean(axis=0)
            std_spectrum = data.std(axis=0)
            
            plt.plot(wavelengths, mean_spectrum, 'b-', label='Mean Spectrum')
            plt.fill_between(wavelengths,
                           mean_spectrum - std_spectrum,
                           mean_spectrum + std_spectrum,
                           alpha=0.2)
            
            plt.xlabel('Wavelength Index')
            plt.ylabel('Reflectance')
            plt.title('Spectral Reflectance Curves')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Spectral curves plot saved to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting spectral curves: {str(e)}")
            raise
    
    def check_data_quality(self, data):
        """Perform comprehensive data quality checks.
        
        Args:
            data (DataFrame): Input data
            
        Returns:
            dict: Dictionary containing quality metrics
        """
        try:
            quality_metrics = {
                'missing_values': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum().sum() / 
                                    (data.shape[0] * data.shape[1]) * 100),
                'zero_values': (data == 0).sum().sum(),
                'negative_values': (data < 0).sum().sum(),
                'samples': data.shape[0],
                'features': data.shape[1]
            }
            
            # Check for potential outliers using IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5 * IQR)) | 
                       (data > (Q3 + 1.5 * IQR))).sum().sum()
            quality_metrics['potential_outliers'] = outliers
            
            # Check for sensor drift
            row_means = data.mean(axis=1)
            drift_z_score = np.abs(stats.zscore(row_means))
            quality_metrics['potential_drift'] = (drift_z_score > 3).sum()
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            raise
    
    def plot_quality_summary(self, data, save_path=None):
        """Create summary plots for data quality visualization.
        
        Args:
            data (DataFrame): Input data
            save_path (str, optional): Path to save the plot
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Distribution of means
            sns.histplot(data=data.mean(axis=1), ax=ax1)
            ax1.set_title('Distribution of Sample Means')
            ax1.set_xlabel('Mean Reflectance')
            
            # Distribution of standard deviations
            sns.histplot(data=data.std(axis=1), ax=ax2)
            ax2.set_title('Distribution of Sample Standard Deviations')
            ax2.set_xlabel('Standard Deviation')
            
            # Boxplot of features
            sns.boxplot(data=data, ax=ax3)
            ax3.set_title('Feature Distributions')
            ax3.set_xlabel('Feature Index')
            ax3.set_ylabel('Value')
            
            # Correlation heatmap of random features
            if data.shape[1] > 10:
                sample_cols = np.random.choice(data.columns, 10, replace=False)
                sns.heatmap(data[sample_cols].corr(), ax=ax4, cmap='coolwarm')
                ax4.set_title('Correlation Heatmap (Sample Features)')
            else:
                sns.heatmap(data.corr(), ax=ax4, cmap='coolwarm')
                ax4.set_title('Correlation Heatmap')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Quality summary plot saved to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting quality summary: {str(e)}")
            raise 