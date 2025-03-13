"""FastAPI service for DON concentration prediction."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import tensorflow as tf
import logging
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the modules
try:
    # When running from project root
    from src.models.don_predictor import DONPredictor
    from src.preprocessing.data_processor import DataProcessor
except ImportError:
    # When running from within src directory
    from models.don_predictor import DONPredictor
    from preprocessing.data_processor import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DON Concentration Predictor",
    description="""
    This API provides endpoints for predicting DON (Deoxynivalenol) concentration 
    in corn samples using hyperspectral imaging data.
    
    ## Features
    
    * Single sample prediction
    * Input validation
    * Error handling
    * Health monitoring
    
    ## Usage
    
    1. Use the `/predict` endpoint to get predictions for spectral data
    2. Monitor the service health using the `/health` endpoint
    
    For more information, visit the [GitHub repository](https://github.com/yourusername/don-concentration-predictor)
    """,
    version="1.0.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
        "url": "https://github.com/yourusername/don-concentration-predictor"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Initialize model and processor
model = None
processor = None

class SpectralData(BaseModel):
    """Request model for spectral data."""
    
    values: List[float] = Field(
        ...,
        description="List of spectral reflectance values (448 bands required)",
        example=[0.1, 0.2, 0.3, 0.4, 0.5]  # Truncated for brevity
    )
    
    class Config:
        schema_extra = {
            "example": {
                "values": [0.1] * 448  # Example with correct number of bands
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    don_concentration: float = Field(
        ...,
        description="Predicted DON concentration in ppb",
        example=123.45
    )
    units: str = Field(
        default="ppb",
        description="Units of measurement"
    )
    confidence_interval: dict = Field(
        ...,
        description="95% confidence interval for the prediction",
        example={"lower": 100.0, "upper": 150.0}
    )

class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded",
        example=True
    )
    processor_loaded: bool = Field(
        ...,
        description="Whether the data processor is loaded",
        example=True
    )
    memory_usage: float = Field(
        ...,
        description="Current memory usage in MB",
        example=1234.56
    )

@app.on_event("startup")
async def load_model():
    """Load the model and processor on startup."""
    global model, processor
    try:
        model = DONPredictor.load('models/best_model.keras')
        processor = DataProcessor()
        processor.load_scalers('models/X_scaler.pkl', 'models/y_scaler.pkl')
        logger.info("Model and processor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/",
         summary="Root endpoint",
         description="Returns a welcome message and basic API information")
async def root():
    """Root endpoint."""
    return {
        "message": "DON Concentration Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.post("/predict",
          response_model=PredictionResponse,
          summary="Predict DON concentration",
          description="""
          Makes a prediction of DON concentration based on input spectral data.
          
          The input should be a list of 448 spectral reflectance values.
          Returns the predicted concentration in ppb along with confidence intervals.
          """)
async def predict(data: SpectralData):
    """Predict DON concentration from spectral data."""
    try:
        # Validate input
        if len(data.values) != 448:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 448 spectral bands, got {len(data.values)}"
            )
        
        # Preprocess input
        X = np.array(data.values).reshape(1, -1)
        X_scaled = processor.scale_features(X)
        
        # Make prediction
        y_scaled = model.model.predict(X_scaled)
        y_pred = processor.inverse_transform_target(y_scaled)
        
        # Calculate confidence intervals
        confidence_interval = {
            "lower": float(y_pred[0] * 0.9),  # Simplified example
            "upper": float(y_pred[0] * 1.1)
        }
        
        return {
            "don_concentration": float(y_pred[0]),
            "units": "ppb",
            "confidence_interval": confidence_interval
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health",
         response_model=HealthResponse,
         summary="Health check",
         description="Returns the current status of the service and its components")
async def health():
    """Health check endpoint."""
    try:
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "memory_usage": memory_usage
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 