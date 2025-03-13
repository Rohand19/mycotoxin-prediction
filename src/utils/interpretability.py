"""Module for model interpretability using SHAP values."""

import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class ModelInterpreter:
    """Class for interpreting model predictions using SHAP values."""
    
    def __init__(self, model: tf.keras.Model):
        """Initialize the interpreter with a trained model."""
        self.model = model
        self.explainer = None
        
    def prepare_explainer(self, background_data: np.ndarray) -> None:
        """
        Prepare the SHAP explainer using background data.
        
        Args:
            background_data: Sample of training data for SHAP explainer
        """
        try:
            self.explainer = shap.DeepExplainer(self.model, background_data)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            raise
            
    def explain_prediction(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Generate SHAP values for input data.
        
        Args:
            input_data: Data to explain predictions for
            
        Returns:
            Tuple of SHAP values and summary statistics
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call prepare_explainer first.")
            
        try:
            shap_values = self.explainer.shap_values(input_data)
            
            # Calculate summary statistics
            summary = {
                "mean_impact": np.mean(np.abs(shap_values), axis=0),
                "max_impact": np.max(np.abs(shap_values), axis=0),
                "top_features": self._get_top_features(shap_values)
            }
            
            return shap_values, summary
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            raise
            
    def plot_feature_importance(self, shap_values: np.ndarray, 
                              feature_names: List[str] = None) -> None:
        """
        Plot feature importance based on SHAP values.
        
        Args:
            shap_values: SHAP values from explain_prediction
            feature_names: Optional list of feature names
        """
        try:
            plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values, feature_names=feature_names)
            plt.title("Feature Importance (SHAP Values)")
            plt.tight_layout()
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
            
    def _get_top_features(self, shap_values: np.ndarray, 
                         top_n: int = 10) -> Dict[int, float]:
        """
        Get the indices of the most important features.
        
        Args:
            shap_values: SHAP values from explain_prediction
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature indices and their importance scores
        """
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:]
        return {int(idx): float(mean_abs_shap[idx]) for idx in top_indices} 