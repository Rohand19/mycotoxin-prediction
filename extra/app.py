from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import logging
from typing import List, Dict
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DON Prediction API",
    description="API for predicting DON concentration in corn samples using hyperspectral data",
    version="1.0.0"
)

# Load model and scalers
try:
    model = tf.keras.models.load_model('don_prediction_model.keras')
    X_scaler = joblib.load('X_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    logger.info("Model and scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scalers: {str(e)}")
    raise

class SpectralData(BaseModel):
    """Request model for spectral data input."""
    features: List[float]

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    don_concentration: float
    confidence_interval: Dict[str, float]

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "DON Prediction API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SpectralData):
    """Predict DON concentration from spectral data.
    
    Args:
        data: SpectralData object containing spectral features
        
    Returns:
        PredictionResponse object with predicted DON concentration
        and confidence interval
    """
    try:
        # Convert input to numpy array
        features = np.array(data.features).reshape(1, -1)
        
        # Validate input dimensions
        if features.shape[1] != 448:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 448 features, got {features.shape[1]}"
            )
        
        # Scale features
        features_scaled = X_scaler.transform(features)
        
        # Make prediction
        prediction_scaled = model.predict(features_scaled)
        
        # Convert back to original scale
        prediction = y_scaler.inverse_transform(prediction_scaled)
        
        # Calculate simple confidence interval (can be enhanced)
        std_dev = np.std(prediction_scaled) * y_scaler.scale_
        confidence_interval = {
            "lower_bound": float(prediction[0][0] - 1.96 * std_dev),
            "upper_bound": float(prediction[0][0] + 1.96 * std_dev)
        }
        
        return PredictionResponse(
            don_concentration=float(prediction[0][0]),
            confidence_interval=confidence_interval
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 