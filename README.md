# DON Concentration Predictor

A machine learning system for predicting DON (Deoxynivalenol) concentration in corn samples using hyperspectral imaging data.

## Overview

This project implements a deep learning pipeline to predict mycotoxin levels (DON concentration) in corn samples using hyperspectral imaging data. The system uses a neural network with an attention mechanism to process spectral reflectance data and make accurate predictions.

## Features

- **Data Processing**
  - Robust data preprocessing with StandardScaler and RobustScaler
  - Automated handling of non-numeric columns
  - Skewness analysis and transformation
  - Memory-efficient processing for large datasets

- **Model Architecture**
  - Neural network with multi-head self-attention mechanism
  - Dimensionality reduction for spectral data
  - Batch normalization and dropout for regularization
  - Configurable hyperparameters via configuration system

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
├── src/                   # Source code
│   ├── api/               # FastAPI service
│   │   └── main.py        # API implementation
│   ├── models/            # Model architecture
│   │   ├── attention.py   # Attention mechanism
│   │   ├── don_predictor.py # Main model
│   │   └── trainer.py     # Training logic
│   ├── preprocessing/     # Data processing
│   │   └── data_processor.py # Data preprocessing
│   ├── utils/             # Utility functions
│   │   ├── data_quality.py  # Data quality checks
│   │   ├── interpretability.py # Model interpretability
│   │   ├── logger.py      # Logging utilities
│   │   ├── metrics.py     # Evaluation metrics
│   │   └── visualization.py # Visualization tools
│   ├── streamlit_app.py   # Streamlit web interface
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

To train the model:

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

To run the Streamlit app:

```bash
streamlit run src/streamlit_app.py
```

The Streamlit app provides:
- Interactive prediction interface
- Model performance visualization
- Data exploration tools
- Feature importance analysis

### API Service

To start the FastAPI service:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t don-predictor .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 don-predictor
```

## API Documentation

The API provides the following endpoints:

- `GET /`: Root endpoint with API information
- `POST /predict`: Make DON concentration predictions
- `GET /health`: Health check with memory usage information

Example prediction request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"values": [0.1, 0.2, ..., 0.5]}'
```

Response format:
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

## Model Architecture

The model uses a neural network with the following components:
- Input layer for 448 spectral bands
- Batch normalization for input stability
- Dimensionality reduction to 256 features
- Multi-head self-attention mechanism
- Multiple dense layers with dropout
- Output layer for DON concentration prediction

## Model Performance

Based on recent training runs, the model achieves:
- R² Score: 0.9058
- RMSE: 5,131.81 ppb
- MAE: 3,295.69 ppb

Performance may vary based on the specific dataset and hyperparameters.

## Troubleshooting

If you encounter import errors:
- Make sure you're running commands from the project root directory
- Check that all dependencies are installed
- If running from within the `src` directory, the code uses relative imports

## Acknowledgments

- Built using TensorFlow, FastAPI, and Streamlit
- Inspired by research in hyperspectral imaging analysis for agricultural applications