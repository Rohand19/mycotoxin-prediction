"""Metrics utilities for DON concentration prediction."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    try:
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Calculate additional statistics
        residuals = y_pred - y_true
        metrics.update({
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'prediction_bias': np.mean(y_pred) - np.mean(y_true)
        })
        
        logger.info("Metrics calculated successfully")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name.upper()}: {value:.4f}")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def calculate_fold_metrics(fold_results):
    """Calculate average metrics across cross-validation folds.
    
    Args:
        fold_results (list): List of dictionaries containing metrics for each fold
        
    Returns:
        dict: Dictionary containing averaged metrics with standard deviations
    """
    try:
        # Initialize dictionaries for means and stds
        metrics_mean = {}
        metrics_std = {}
        
        # Get all metric names from the first fold
        metric_names = fold_results[0].keys()
        
        # Calculate mean and std for each metric
        for metric in metric_names:
            values = [fold[metric] for fold in fold_results]
            metrics_mean[f"{metric}_mean"] = np.mean(values)
            metrics_std[f"{metric}_std"] = np.std(values)
        
        # Combine means and stds
        combined_metrics = {**metrics_mean, **metrics_std}
        
        logger.info("Cross-validation metrics calculated successfully")
        for metric_name, value in combined_metrics.items():
            logger.info(f"{metric_name.upper()}: {value:.4f}")
            
        return combined_metrics
        
    except Exception as e:
        logger.error(f"Error calculating cross-validation metrics: {str(e)}")
        raise

def calculate_confidence_intervals(y_true, y_pred, confidence_level=0.95):
    """Calculate confidence intervals for predictions.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        confidence_level (float): Confidence level (default: 0.95)
        
    Returns:
        dict: Dictionary containing confidence intervals
    """
    try:
        residuals = y_pred - y_true
        std_error = np.std(residuals)
        
        # Calculate z-score for the given confidence level
        z_score = np.abs(np.percentile(np.random.standard_normal(10000),
                                     (1 - confidence_level) * 100 / 2))
        
        confidence_intervals = {
            'lower_bound': y_pred - z_score * std_error,
            'upper_bound': y_pred + z_score * std_error,
            'std_error': std_error,
            'confidence_level': confidence_level
        }
        
        logger.info(f"Confidence intervals calculated at {confidence_level*100}% level")
        logger.info(f"Standard Error: {std_error:.4f}")
        
        return confidence_intervals
        
    except Exception as e:
        logger.error(f"Error calculating confidence intervals: {str(e)}")
        raise 