"""Visualization utilities for DON concentration prediction."""

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def set_style():
    """Set the plotting style."""
    plt.style.use('default')  # Using default matplotlib style

def plot_training_history(history, save_path=None):
    """Plot training history and save the figure."""
    try:
        set_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def plot_predictions(y_true, y_pred, save_path=None):
    """Plot actual vs predicted values.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        save_path (str, optional): Path to save the plot
    """
    try:
        set_style()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'r--', linewidth=2)
        ax1.set_xlabel('Actual DON Concentration (ppb)')
        ax1.set_ylabel('Predicted DON Concentration (ppb)')
        ax1.set_title('Actual vs. Predicted Values')
        ax1.grid(True)
        
        # Residuals plot
        residuals = y_pred - y_true
        ax2.hist(residuals, bins=20, density=True, alpha=0.5, label='Residuals')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals (ppb)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Residuals')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")
        raise

def plot_feature_importance(feature_importance, feature_names, save_path=None):
    """Plot feature importance.
    
    Args:
        feature_importance (array-like): Feature importance scores
        feature_names (list): Feature names
        save_path (str, optional): Path to save the plot
    """
    try:
        set_style()
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[-20:]  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)),
                feature_importance[indices],
                align='center')
        plt.yticks(range(len(indices)),
                  [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance Score')
        plt.title('Top 20 Most Important Features')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def plot_attention_weights(attention_weights, save_path=None):
    """Plot attention weights heatmap.
    
    Args:
        attention_weights (array-like): Attention weights matrix
        save_path (str, optional): Path to save the plot
    """
    try:
        set_style()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(attention_weights, cmap='viridis', aspect='auto')
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Key Dimension')
        plt.ylabel('Query Dimension')
        plt.colorbar(label='Attention Weight')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Attention weights plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting attention weights: {str(e)}")
        raise 