"""FastAPI service for DON concentration prediction."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import logging
import sys
import os
import joblib
import psutil

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the modules
try:
    # When running from project root
    from src.models.simple_predictor import SimplePredictor
except ImportError:
    # When running from within src directory
    from models.simple_predictor import SimplePredictor

# Get configuration from environment variables
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/rf_model_real_data.joblib')
X_SCALER_PATH = os.environ.get('X_SCALER_PATH', 'models/X_scaler_real.pkl')
Y_SCALER_PATH = os.environ.get('Y_SCALER_PATH', 'models/y_scaler_real.pkl')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    """,
    version="1.0.0"
)

# Initialize model
model = None

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
    confidence_interval: Optional[Dict[str, float]] = Field(
        None,
        description="Confidence interval for the prediction (if available)",
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
    memory_usage: float = Field(
        ...,
        description="Current memory usage in MB",
        example=1234.56
    )
    model_type: str = Field(
        ...,
        description="Type of model being used",
        example="RandomForest"
    )

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        logger.info(f"Loading X scaler from {X_SCALER_PATH}")
        logger.info(f"Loading Y scaler from {Y_SCALER_PATH}")
        
        # Check if model files exist
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
        if not os.path.exists(X_SCALER_PATH):
            logger.error(f"X scaler file not found: {X_SCALER_PATH}")
            raise FileNotFoundError(f"X scaler file not found: {X_SCALER_PATH}")
            
        if not os.path.exists(Y_SCALER_PATH):
            logger.error(f"Y scaler file not found: {Y_SCALER_PATH}")
            raise FileNotFoundError(f"Y scaler file not found: {Y_SCALER_PATH}")
        
        # Load the scikit-learn model
        model = SimplePredictor()
        model.model = joblib.load(MODEL_PATH)
        
        # Load the scalers
        model.X_scaler = joblib.load(X_SCALER_PATH)
        model.y_scaler = joblib.load(Y_SCALER_PATH)
        
        logger.info("Model and scalers loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Don't raise here to allow the API to start even if model loading fails
        # The health endpoint will report the issue

@app.get("/",
         summary="Root endpoint",
         description="Returns a welcome message and basic API information")
async def root():
    """Root endpoint."""
    return {
        "message": "DON Concentration Prediction API (scikit-learn version)",
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
          Returns the predicted concentration in ppb.
          """)
async def predict(data: SpectralData):
    """Predict DON concentration from spectral data."""
    try:
        # Check if model is loaded
        if model is None or model.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
            
        # Validate input
        if len(data.values) != 448:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 448 spectral bands, got {len(data.values)}"
            )
        
        # Check for non-numeric values
        if not all(isinstance(x, (int, float)) for x in data.values):
            raise HTTPException(
                status_code=400,
                detail="All values must be numeric"
            )
        
        # Preprocess input and make prediction using the model's internal scaling
        X = np.array(data.values).reshape(1, -1)
        
        try:
            y_scaled = model.predict(X)
            y_pred = model.inverse_transform_target(y_scaled)
            
            return {
                "don_concentration": float(y_pred[0][0]),
                "units": "ppb"
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health",
         response_model=HealthResponse,
         summary="Health check",
         description="Returns the current status of the service and its components")
async def health():
    """Health check endpoint."""
    try:
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        model_loaded = model is not None and model.model is not None
        model_type = "RandomForest" if model_loaded else "None"
        
        if model_loaded:
            try:
                # Try to get more specific model type
                model_type = model.model.__class__.__name__
            except:
                pass
        
        return {
            "status": "healthy" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "memory_usage": memory_usage,
            "model_type": model_type
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 