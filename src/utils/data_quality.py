"""Data quality analysis and visualization utilities."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Class for analyzing and visualizing data quality."""

    def __init__(self):
        """Initialize the analyzer."""
        plt.style.use("default")

    def plot_spectral_curves(self, data, wavelengths=None, save_path=None):
        """Plot spectral reflectance curves.

        Args:
            data (DataFrame): Spectral data
            wavelengths (array-like, optional): Wavelength values
            save_path (str, optional): Path to save the plot
        """
        try:
            # First, ensure we're only working with numeric data
            numeric_data = data.select_dtypes(include=["number"])

            if numeric_data.empty:
                logger.warning("No numeric columns found in the data for plotting spectral curves")
                # Create a simple figure with a text message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5,
                    0.5,
                    "No numeric data available for plotting spectral curves",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.axis("off")
                return

            plt.figure(figsize=(12, 6))

            # For spectral data, we typically want to plot each column (wavelength) on the x-axis
            # and the reflectance values on the y-axis

            # Get column names or indices for x-axis
            if wavelengths is None:
                # Use column indices or names as wavelengths
                wavelengths = list(range(len(numeric_data.columns)))

            # Calculate mean and std across samples (rows)
            mean_values = numeric_data.mean(axis=0).values
            std_values = numeric_data.std(axis=0).values

            # Plot the mean curve
            plt.plot(
                wavelengths[: len(mean_values)],
                mean_values,
                "b-",
                label="Mean Spectrum",
            )

            # Add confidence interval (mean ± std)
            plt.fill_between(
                wavelengths[: len(mean_values)],
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.2,
            )

            # Plot a few individual samples for reference (up to 5)
            num_samples = min(5, numeric_data.shape[0])
            for i in range(num_samples):
                sample_values = numeric_data.iloc[i].values
                plt.plot(
                    wavelengths[: len(sample_values)],
                    sample_values,
                    "k-",
                    alpha=0.1,
                )

            plt.xlabel("Spectral Band Index")
            plt.ylabel("Reflectance Value")
            plt.title("Spectral Reflectance Curves")
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
            # First, ensure we're only working with numeric data
            numeric_data = data.select_dtypes(include=["number"])

            if numeric_data.empty:
                logger.warning("No numeric columns found in the data")
                return {
                    "error": "No numeric columns found",
                    "samples": data.shape[0],
                    "features": data.shape[1],
                    "numeric_features": 0,
                }

            # Log information about non-numeric columns
            non_numeric_cols = [col for col in data.columns if col not in numeric_data.columns]
            if non_numeric_cols:
                logger.info(f"Excluding non-numeric columns from analysis: {', '.join(non_numeric_cols)}")

            quality_metrics = {
                "missing_values": numeric_data.isnull().sum().sum(),
                "missing_percentage": (
                    numeric_data.isnull().sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1]) * 100
                ),
                "zero_values": (numeric_data == 0).sum().sum(),
                "negative_values": (numeric_data < 0).sum().sum(),
                "samples": numeric_data.shape[0],
                "features": numeric_data.shape[1],
                "total_features": data.shape[1],
                "non_numeric_features": len(non_numeric_cols),
            }

            # Check for potential outliers using IQR
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().sum()
            quality_metrics["potential_outliers"] = outliers

            # Check for sensor drift
            row_means = numeric_data.mean(axis=1)
            drift_z_score = np.abs(stats.zscore(row_means))
            quality_metrics["potential_drift"] = (drift_z_score > 3).sum()

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
            # First, ensure we're only working with numeric data
            numeric_data = data.select_dtypes(include=["number"])

            if numeric_data.empty:
                logger.warning("No numeric columns found in the data for plotting")
                # Create a simple figure with a text message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5,
                    0.5,
                    "No numeric data available for plotting",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.axis("off")
                return

            # Log information about non-numeric columns
            non_numeric_cols = [col for col in data.columns if col not in numeric_data.columns]
            if non_numeric_cols:
                logger.info(f"Excluding non-numeric columns from plotting: {', '.join(non_numeric_cols)}")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Distribution of row means (mean value across all bands for each sample)
            row_means = numeric_data.mean(axis=1)
            sns.histplot(row_means, ax=ax1, kde=True)
            ax1.set_title("Distribution of Sample Means")
            ax1.set_xlabel("Mean Value")
            ax1.set_ylabel("Count")

            # 2. Distribution of row standard deviations
            row_stds = numeric_data.std(axis=1)
            sns.histplot(row_stds, ax=ax2, kde=True)
            ax2.set_title("Distribution of Sample Standard Deviations")
            ax2.set_xlabel("Standard Deviation")
            ax2.set_ylabel("Count")

            # 3. Boxplot of a sample of columns (spectral bands)
            # Select a subset of columns if there are too many
            if numeric_data.shape[1] > 20:
                # Take evenly spaced columns for better representation
                indices = np.linspace(0, numeric_data.shape[1] - 1, 20, dtype=int)
                sample_cols = numeric_data.columns[indices]
                plot_data = numeric_data[sample_cols]
                title = "Distribution of 20 Sampled Spectral Bands"
            else:
                plot_data = numeric_data
                title = "Distribution of All Spectral Bands"

            # Create a melted dataframe for better boxplot
            melted_data = pd.melt(
                plot_data.reset_index(),
                id_vars=["index"],
                value_vars=plot_data.columns,
            )
            sns.boxplot(x="variable", y="value", data=melted_data, ax=ax3)
            ax3.set_title(title)
            ax3.set_xlabel("Spectral Band")
            ax3.set_ylabel("Value")
            ax3.tick_params(axis="x", rotation=90)  # Rotate x labels for readability

            # 4. Heatmap of correlations between bands
            # Select a subset of columns if there are too many
            if numeric_data.shape[1] > 10:
                # Take evenly spaced columns for better representation
                indices = np.linspace(0, numeric_data.shape[1] - 1, 10, dtype=int)
                sample_cols = numeric_data.columns[indices]
                corr_data = numeric_data[sample_cols].corr()
                title = "Correlation Between 10 Sampled Spectral Bands"
            else:
                corr_data = numeric_data.corr()
                title = "Correlation Between All Spectral Bands"

            # Plot the correlation heatmap
            sns.heatmap(
                corr_data,
                ax=ax4,
                cmap="coolwarm",
                annot=False,
                vmin=-1,
                vmax=1,
                center=0,
            )
            ax4.set_title(title)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Quality summary plot saved to {save_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting quality summary: {str(e)}")
            raise
