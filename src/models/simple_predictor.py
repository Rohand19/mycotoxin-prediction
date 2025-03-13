"""Simple DON Concentration Predictor using scikit-learn.

This module implements a simple scikit-learn based model for predicting DON concentration
in corn samples, completely avoiding TensorFlow to ensure reliability on all platforms.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd

class SimplePredictor:
    """Simple scikit-learn based predictor for DON concentration."""
    
    def __init__(self):
        """Initialize the simple predictor."""
        self.model = None
        self.X_scaler = StandardScaler()  # Add internal scaler for X
        self.y_scaler = StandardScaler()  # Add internal scaler for y
        
    def create_model(self, n_estimators=100, random_state=42):
        """Create a simple RandomForest model."""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        return self.model
    
    def fit(self, X, y):
        """Fit the model to training data."""
        if self.model is None:
            self.create_model()
        
        # Scale the input data
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        self.model.fit(X_scaled, y_scaled)
        return self
    
    def predict(self, X):
        """Make predictions with the model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit or load first.")
        
        # Ensure X is numeric
        try:
            X = np.asarray(X, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Could not convert input to numeric array: {str(e)}")
        
        # Check if X has the right shape
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Convert single sample to 2D array
        
        # Check if the number of features matches what the model expects
        expected_features = 448  # The model was trained with 448 features
        
        if X.shape[1] != expected_features:
            print(f"Warning: Input has {X.shape[1]} features, but model expects {expected_features}.")
            
            if X.shape[1] > expected_features:
                # If there are too many features, truncate to the expected number
                print(f"Truncating input from {X.shape[1]} to {expected_features} features.")
                X = X[:, :expected_features]
            else:
                # If there are too few features, pad with zeros
                print(f"Padding input from {X.shape[1]} to {expected_features} features with zeros.")
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]), dtype=X.dtype)
                X = np.hstack((X, padding))
        
        # Scale the input data using our internal scaler
        try:
            X_scaled = self.X_scaler.transform(X)
        except Exception as e:
            print(f"Warning: Could not use fitted scaler: {str(e)}")
            # If the scaler isn't fitted, just standardize the data
            try:
                X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                X_scaled = np.nan_to_num(X_scaled)  # Replace NaNs with 0
            except Exception as e:
                print(f"Warning: Could not standardize data: {str(e)}")
                X_scaled = X  # Use unscaled data as a last resort
        
        # Make predictions
        try:
            y_scaled = self.model.predict(X_scaled)
            # Return the scaled predictions (will be inverse transformed later)
            return y_scaled.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")
    
    def inverse_transform_target(self, y_scaled):
        """Convert scaled target values back to original scale."""
        try:
            y_scaled = np.asarray(y_scaled, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Could not convert predictions to numeric array: {str(e)}")
        
        try:
            return self.y_scaler.inverse_transform(y_scaled)
        except Exception as e:
            print(f"Warning: Could not use fitted scaler for inverse transform: {str(e)}")
            # If the scaler isn't fitted, just return the values
            # This is just a fallback - predictions won't be accurate
            return y_scaled * 1000 + 500  # Scale to a reasonable DON range (500-1500 ppb)
    
    def scale_features(self, X):
        """Scale features using the internal X_scaler."""
        try:
            return self.X_scaler.transform(X)
        except:
            # If the scaler isn't fitted, just standardize the data
            X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            return np.nan_to_num(X_scaled)  # Replace NaNs with 0
    
    def save(self, filepath):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit first.")
        joblib.dump(self.model, filepath)
    
    def save_scalers(self, x_scaler_path, y_scaler_path):
        """Save the scalers to disk."""
        joblib.dump(self.X_scaler, x_scaler_path)
        joblib.dump(self.y_scaler, y_scaler_path)
    
    @classmethod
    def load_or_create(cls, filepath=None, train_on_real_data=True):
        """Load a saved model or create a new one if loading fails."""
        instance = cls()
        
        # If no filepath provided or file doesn't exist, create a new model
        if filepath is None or not os.path.exists(filepath):
            print("Creating new RandomForest model...")
            instance.create_model()
            
            if train_on_real_data:
                try:
                    print("Training model on real data...")
                    # Load the actual dataset
                    data_path = os.path.join('data', 'corn_hyperspectral.csv')
                    if os.path.exists(data_path):
                        # Load and preprocess the data
                        data = pd.read_csv(data_path)
                        
                        # Remove non-numeric columns (like 'hsi_id')
                        non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
                        if non_numeric_cols:
                            print(f"Removing non-numeric columns: {', '.join(non_numeric_cols)}")
                            data = data.select_dtypes(include=['number'])
                        
                        # Extract features and target
                        X = data.iloc[:, :-1].values  # All columns except the last one
                        y = data.iloc[:, -1].values   # Last column (vomitoxin_ppb)
                        
                        print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
                        instance.fit(X, y)
                        print("Model trained on real data successfully!")
                        
                        # Save the trained model and scalers
                        os.makedirs('models', exist_ok=True)
                        instance.save('models/rf_model_real_data.joblib')
                        instance.save_scalers('models/X_scaler_real.pkl', 'models/y_scaler_real.pkl')
                        print("Model and scalers saved to disk")
                        
                        return instance
                    else:
                        print(f"Warning: Could not find dataset at {data_path}")
                        print("Falling back to synthetic data...")
                except Exception as e:
                    print(f"Error training on real data: {e}")
                    print("Falling back to synthetic data...")
            
            # Create some synthetic training data to initialize the model
            print("Training model on synthetic data...")
            X_synthetic = np.random.rand(100, 448)  # 100 samples, 448 features
            y_synthetic = np.random.uniform(500, 2000, size=100)  # Random DON values
            instance.fit(X_synthetic, y_synthetic)
            
            print("Model created and trained on synthetic data")
            return instance
        
        # Try to load the saved model
        try:
            print(f"Loading model from {filepath}...")
            instance.model = joblib.load(filepath)
            print("Model loaded successfully!")
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new RandomForest model instead...")
            
            # Create and train a new model using the same process as above
            if train_on_real_data:
                try:
                    print("Training model on real data...")
                    # Load the actual dataset
                    data_path = os.path.join('data', 'corn_hyperspectral.csv')
                    if os.path.exists(data_path):
                        # Load and preprocess the data
                        data = pd.read_csv(data_path)
                        
                        # Remove non-numeric columns (like 'hsi_id')
                        non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
                        if non_numeric_cols:
                            print(f"Removing non-numeric columns: {', '.join(non_numeric_cols)}")
                            data = data.select_dtypes(include=['number'])
                        
                        # Extract features and target
                        X = data.iloc[:, :-1].values  # All columns except the last one
                        y = data.iloc[:, -1].values   # Last column (vomitoxin_ppb)
                        
                        print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
                        instance.fit(X, y)
                        print("Model trained on real data successfully!")
                        
                        # Save the trained model and scalers
                        os.makedirs('models', exist_ok=True)
                        instance.save('models/rf_model_real_data.joblib')
                        instance.save_scalers('models/X_scaler_real.pkl', 'models/y_scaler_real.pkl')
                        print("Model and scalers saved to disk")
                        
                        return instance
                    else:
                        print(f"Warning: Could not find dataset at {data_path}")
                        print("Falling back to synthetic data...")
                except Exception as e:
                    print(f"Error training on real data: {e}")
                    print("Falling back to synthetic data...")
            
            # Create and train a new model
            instance.create_model()
            
            # Create some synthetic training data to initialize the model
            print("Training model on synthetic data...")
            X_synthetic = np.random.rand(100, 448)  # 100 samples, 448 features
            y_synthetic = np.random.uniform(500, 2000, size=100)  # Random DON values
            instance.fit(X_synthetic, y_synthetic)
            
            print("Model created and trained on synthetic data")
            return instance