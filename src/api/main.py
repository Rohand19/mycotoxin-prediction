"""FastAPI service for DON concentration prediction."""

import logging
import os
import sys
from typing import Dict, List, Optional

import joblib
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now import the modules
try:
    # When running from project root
    from src.models.don_predictor import DONPredictor
    from src.models.simple_predictor import SimplePredictor
except ImportError:
    # When running from within src directory
    from models.don_predictor import DONPredictor
    from models.simple_predictor import SimplePredictor

# Get configuration from environment variables
MODEL_TYPE = os.environ.get("MODEL_TYPE", "tensorflow")  # 'tensorflow' or 'randomforest'

# Model paths based on type
if MODEL_TYPE == "tensorflow":
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.keras")
    X_SCALER_PATH = os.environ.get("X_SCALER_PATH", "models/X_scaler.pkl")
    Y_SCALER_PATH = os.environ.get("Y_SCALER_PATH", "models/y_scaler.pkl")
else:
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/rf_model.joblib")
    X_SCALER_PATH = os.environ.get("X_SCALER_PATH", "models/X_scaler_rf.pkl")
    Y_SCALER_PATH = os.environ.get("Y_SCALER_PATH", "models/y_scaler_rf.pkl")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DON Concentration Predictor",
    description="""
    This API provides endpoints for predicting DON (Deoxynivalenol) concentration
    in corn samples using hyperspectral imaging data.

    ## Features

    * Single sample prediction using TensorFlow or RandomForest models
    * Input validation
    * Error handling
    * Health monitoring

    ## Model Selection

    The API can be configured to use either:
    * TensorFlow model with attention mechanism (default)
    * RandomForest model for faster inference or Apple Silicon compatibility

    Set the MODEL_TYPE environment variable to 'tensorflow' or 'randomforest'
    to choose the implementation.
    """,
    version="1.0.0",
)

# Initialize model
model = None


class SpectralData(BaseModel):
    """Request model for spectral data."""

    values: List[float] = Field(
        ...,
        description="List of spectral reflectance values (448 bands required)",
        example=[0.1, 0.2, 0.3, 0.4, 0.5],  # Truncated for brevity
    )

    class Config:
        schema_extra = {"example": {"values": [0.1] * 448}}  # Example with correct number of bands


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    don_concentration: float = Field(..., description="Predicted DON concentration in ppb", example=123.45)
    units: str = Field(default="ppb", description="Units of measurement")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None,
        description="Confidence interval for the prediction (if available)",
        example={"lower": 100.0, "upper": 150.0},
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status", example="healthy")
    model_loaded: bool = Field(..., description="Whether the model is loaded", example=True)
    memory_usage: float = Field(..., description="Current memory usage in MB", example=1234.56)
    model_type: str = Field(..., description="Type of model being used", example="TensorFlow")


@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    try:
        logger.info(f"Loading {MODEL_TYPE} model from {MODEL_PATH}")
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

        if MODEL_TYPE == "tensorflow":
            # Load TensorFlow model
            model = DONPredictor(input_shape=448)
            model = DONPredictor.load(MODEL_PATH)
            # Load scalers
            model.X_scaler = joblib.load(X_SCALER_PATH)
            model.y_scaler = joblib.load(Y_SCALER_PATH)
        else:
            # Load RandomForest model
            model = SimplePredictor()
            model.model = joblib.load(MODEL_PATH)
            # Load scalers
            model.X_scaler = joblib.load(X_SCALER_PATH)
            model.y_scaler = joblib.load(Y_SCALER_PATH)

        logger.info(f"{MODEL_TYPE} model and scalers loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.get(
    "/",
    summary="Root endpoint",
    description="Returns a welcome message and basic API information",
)
async def root():
    """Root endpoint."""
    return {
        "message": f"DON Concentration Prediction API ({MODEL_TYPE} version)",
        "version": "1.0.0",
        "model_type": MODEL_TYPE,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict DON concentration",
    description="""
    Makes a prediction of DON concentration based on input spectral data.
    
    The input should be a list of 448 spectral reflectance values.
    Returns the predicted concentration in ppb.
    
    When using the TensorFlow model, confidence intervals are provided.
    The RandomForest model provides point estimates only.
    """,
)
async def predict(data: SpectralData):
    """Predict DON concentration from spectral data."""
    try:
        # Check if model is loaded
        if model is None or (hasattr(model, 'model') and model.model is None):
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs.",
            )

        # Validate input
        if len(data.values) != 448:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 448 spectral bands, got {len(data.values)}",
            )

        # Check for non-numeric values
        if not all(isinstance(x, (int, float)) for x in data.values):
            raise HTTPException(status_code=400, detail="All values must be numeric")

        # Convert input to numpy array with proper dtype
        X = np.array(data.values, dtype=np.float32, copy=True).reshape(1, -1)

        try:
            if MODEL_TYPE == "tensorflow":
                # Scale features
                X_scaled = model.X_scaler.transform(X)
                # Make prediction
                y_scaled = model.model.predict(X_scaled)
                # Convert back to original scale
                y_pred = model.y_scaler.inverse_transform(y_scaled)
                
                # Calculate confidence interval (simplified example)
                pred_value = float(y_pred[0][0])
                confidence = 0.1 * pred_value  # 10% confidence interval
                
                return {
                    "don_concentration": pred_value,
                    "units": "ppb",
                    "confidence_interval": {
                        "lower": pred_value - confidence,
                        "upper": pred_value + confidence
                    }
                }
            else:
                # Use RandomForest model's predict method
                y_scaled = model.predict(X)
                y_pred = model.inverse_transform_target(y_scaled)
                
                return {
                    "don_concentration": float(y_pred[0][0]),
                    "units": "ppb"
                }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current status of the service and its components",
)
async def health():
    """Health check endpoint."""
    try:
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB

        model_loaded = model is not None and (
            (MODEL_TYPE == "tensorflow" and model.model is not None) or
            (MODEL_TYPE == "randomforest" and model.model is not None)
        )

        return {
            "status": "healthy" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "memory_usage": memory_usage,
            "model_type": MODEL_TYPE.capitalize(),
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
