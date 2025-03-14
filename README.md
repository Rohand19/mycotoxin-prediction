# DON Concentration Predictor

A machine learning system for predicting DON (Deoxynivalenol) concentration in corn samples using hyperspectral imaging data.

## Overview

This project implements a deep learning pipeline to predict mycotoxin levels (DON concentration) in corn samples using hyperspectral imaging data. The primary implementation uses a TensorFlow-based neural network with an attention mechanism to process spectral reflectance data and make accurate predictions. An alternative scikit-learn RandomForest implementation is also provided for environments where TensorFlow may not be optimal.

## Features

- **Data Processing**
  - Robust data preprocessing with StandardScaler and RobustScaler
  - Automated handling of non-numeric columns
  - Skewness analysis and transformation
  - Memory-efficient processing for large datasets

- **Primary Model Architecture (TensorFlow)**
  - Neural network with multi-head self-attention mechanism
  - Dimensionality reduction for spectral data
  - Batch normalization and dropout for regularization
  - Configurable hyperparameters via configuration system

- **Alternative Model (scikit-learn)**
  - RandomForest regressor for environments where TensorFlow may not be optimal
  - Simplified preprocessing pipeline
  - Comparable performance with lower computational requirements
  - Particularly useful for Apple Silicon Macs and environments without GPU support

- **Evaluation & Metrics**
  - Multiple evaluation metrics (MAE, RMSE, R², Mean/Std Residual)
  - Visualization of predictions vs. actual values
  - Training history visualization
  - Model interpretability with SHAP values

- **Production Features**
  - FastAPI service for real-time predictions with confidence intervals
  - Streamlit web interface for interactive use
  - Docker containerization
  - Comprehensive logging and memory usage tracking

## Project Structure

```
.
├── data/                  # Data directory
│   └── corn_hyperspectral.csv
├── models/                # Saved models and scalers
│   ├── best_model.keras   # Primary TensorFlow model
│   ├── X_scaler.pkl       # Feature scaler for TensorFlow model
│   ├── y_scaler.pkl       # Target scaler for TensorFlow model
│   ├── rf_model_real_data.joblib  # Alternative RandomForest model
│   ├── X_scaler_real.pkl  # Feature scaler for RandomForest model
│   └── y_scaler_real.pkl  # Target scaler for RandomForest model
├── src/                   # Source code
│   ├── api/               # FastAPI service
│   │   └── main.py        # API implementation
│   ├── models/            # Model architecture
│   │   ├── attention.py   # Attention mechanism for TensorFlow model
│   │   ├── don_predictor.py # Primary TensorFlow model
│   │   ├── simple_predictor.py # Alternative RandomForest model
│   │   └── trainer.py     # Training logic
│   ├── preprocessing/     # Data processing
│   │   └── data_processor.py # Data preprocessing
│   ├── utils/             # Utility functions
│   │   ├── data_quality.py  # Data quality checks
│   │   ├── interpretability.py # Model interpretability
│   │   ├── logger.py      # Logging utilities
│   │   ├── metrics.py     # Evaluation metrics
│   │   └── visualization.py # Visualization tools
│   ├── streamlit_app.py   # Primary Streamlit web interface (TensorFlow)
│   ├── streamlit_app_simple_sklearn.py # Alternative Streamlit interface (RandomForest)
│   └── train.py           # Training script
├── tests/                 # Unit tests
│   ├── test_model.py      # Model tests
│   └── test_integration.py # Integration tests
├── visualizations/        # Generated plots
├── Dockerfile             # Docker configuration
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rohand19/don-concentration-predictor.git
cd don-concentration-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the primary TensorFlow model:

```bash
python src/train.py
```

This will:
- Load and preprocess the hyperspectral data
- Train the DON prediction model
- Save the best model and scalers
- Generate performance visualizations

### Running Tests

To run the tests:

```bash
python run_tests.py
```

### Web Interface

To run the primary Streamlit app (TensorFlow-based):

```bash
streamlit run src/streamlit_app.py
```

To run the alternative Streamlit app (RandomForest-based, useful for Apple Silicon Macs):

```bash
streamlit run src/streamlit_app_simple_sklearn.py
```

The Streamlit apps provide:
- Interactive prediction interface
- Model performance visualization
- Data exploration tools
- Feature importance analysis

### API Service

To start the FastAPI service:

```bash
# Using TensorFlow model (default):
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Using RandomForest model:
MODEL_TYPE=randomforest uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API supports both TensorFlow and RandomForest models:
- TensorFlow model provides confidence intervals and higher accuracy
- RandomForest model offers faster inference and better compatibility with Apple Silicon

Configuration via environment variables:
- `MODEL_TYPE`: Set to 'tensorflow' (default) or 'randomforest'
- `MODEL_PATH`: Path to the model file (defaults to appropriate model based on type)
- `X_SCALER_PATH`: Path to feature scaler
- `Y_SCALER_PATH`: Path to target scaler
- `LOG_LEVEL`: Logging level (default: INFO)

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t don-predictor .
```

2. Run the container:
```bash
# Using TensorFlow model:
docker run -p 8000:8000 -p 8501:8501 don-predictor

# Using RandomForest model:
docker run -p 8000:8000 -p 8501:8501 -e MODEL_TYPE=randomforest don-predictor
```

## API Documentation

The API provides the following endpoints:

- `GET /`: Root endpoint with API information and model type
- `POST /predict`: Make DON concentration predictions
- `GET /health`: Health check with memory usage and model status

Example prediction request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"values": [0.1, 0.2, ..., 0.5]}'  # 448 spectral values required
```

Response format (TensorFlow model):
```json
{
  "don_concentration": 123.45,
  "units": "ppb",
  "confidence_interval": {
    "lower": 100.0,
    "upper": 150.0
  }
}
```

Response format (RandomForest model):
```json
{
  "don_concentration": 123.45,
  "units": "ppb"
}
```

Health check response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory_usage": 1234.56,
  "model_type": "TensorFlow"  // or "RandomForest"
}
```

## Model Architecture

### Primary Model (TensorFlow)

The primary model uses a neural network with the following components:
- Input layer for 448 spectral bands
- Batch normalization for input stability
- Dimensionality reduction to 256 features
- Multi-head self-attention mechanism
- Multiple dense layers with dropout
- Output layer for DON concentration prediction

### Alternative Model (RandomForest)

The alternative model uses scikit-learn's RandomForest regressor:
- Trained on the same 448 spectral bands
- Internal feature importance calculation
- Robust to outliers and non-linear relationships
- Lower computational requirements than the neural network
- Particularly useful for environments where TensorFlow may not be optimal

## Model Performance

Based on recent training runs, the models achieve:

**TensorFlow Model:**
- R² Score: 0.9058
- RMSE: 5,131.81 ppb
- MAE: 3,295.69 ppb

**RandomForest Model:**
- R² Score: 0.8923
- RMSE: 5,487.32 ppb
- MAE: 3,412.45 ppb

Performance may vary based on the specific dataset and hyperparameters.

## Choosing Between Models

- **Use the TensorFlow model when:**
  - You have GPU acceleration available
  - Maximum accuracy is required
  - You're working on a system with good TensorFlow support

- **Use the RandomForest model when:**
  - You're working on Apple Silicon Macs or systems with limited TensorFlow support
  - You need faster inference with comparable accuracy
  - You prefer a simpler model with built-in feature importance

## Troubleshooting

If you encounter import errors:
- Make sure you're running commands from the project root directory
- Check that all dependencies are installed
- If running from within the `src` directory, the code uses relative imports

If you encounter TensorFlow issues on Apple Silicon Macs:
- Try using the alternative RandomForest implementation
- Run `streamlit run src/streamlit_app_simple_sklearn.py` instead of the default app

## Acknowledgments

- Built using TensorFlow, scikit-learn, FastAPI, and Streamlit
- Built using Cursor IDE
