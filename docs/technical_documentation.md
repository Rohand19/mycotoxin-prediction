# Technical Documentation

## System Architecture

The DON Concentration Predictor is built with a modular architecture consisting of several key components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Pipeline  │────▶│  Model Training │────▶│  Evaluation     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  FastAPI        │     │  Streamlit      │     │  Visualization  │
│  Service        │     │  Interface      │     │  Tools          │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

The system offers two implementation approaches:
1. **Primary Implementation**: TensorFlow-based neural network with attention mechanism
2. **Alternative Implementation**: scikit-learn RandomForest regressor

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
├── docs/                  # Documentation
│   ├── technical_documentation.md  # This file
│   ├── api_documentation.md       # API documentation
│   └── streamlit_user_guide.md    # Streamlit interface guide
├── Dockerfile             # Docker configuration
├── README.md             # Project overview
├── REPORT.md             # Technical report
└── requirements.txt      # Dependencies
```

## Component Details

### 1. Data Pipeline (`src/preprocessing/data_processor.py`)
- **Data Loading**: Reads hyperspectral data from CSV files
- **Preprocessing**:
  - Wavelength normalization
  - Outlier detection and removal
  - Feature scaling using StandardScaler
  - Train-test split with stratification
- **Data Quality Checks**:
  - Missing value detection
  - Spectral range validation
  - Data distribution analysis

### 2. Model Architecture
#### Primary Implementation (`src/models/don_predictor.py`)
- **Neural Network Architecture**:
  - Input layer: Matches spectral dimensions
  - Attention mechanism for feature importance
  - Dense layers with dropout for regularization
  - Output layer for DON concentration prediction
- **Training Parameters**:
  - Optimizer: Adam
  - Loss function: Mean Squared Error
  - Learning rate: 0.001
  - Batch size: 32
  - Epochs: 100 with early stopping

#### Alternative Implementation (`src/models/simple_predictor.py`)
- **RandomForest Regressor**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 2
  - min_samples_leaf: 1
- **Advantages**:
  - Faster training and inference
  - Better performance on Apple Silicon
  - No GPU requirements

### 3. API Service (`src/api/main.py`)
- **FastAPI Implementation**:
  - Health check endpoint
  - Prediction endpoint with JSON input/output
  - Model switching capability
  - Error handling and validation
- **Deployment**:
  - Docker containerization
  - Environment variable configuration
  - Load balancing support

### 4. Web Interface
#### Primary Interface (`src/streamlit_app.py`)
- **Features**:
  - Single sample prediction
  - Batch prediction support
  - Interactive visualization
  - Model performance metrics
  - Attention weights visualization

#### Alternative Interface (`src/streamlit_app_simple_sklearn.py`)
- **Features**:
  - Simplified prediction interface
  - Feature importance plots
  - Performance comparison
  - Cross-validation results

### 5. Evaluation (`src/utils/metrics.py`)
- **Metrics**:
  - R-squared (R²)
  - Root Mean Square Error (RMSE)
  - Mean Absolute Error (MAE)
  - Prediction intervals
- **Validation**:
  - K-fold cross-validation
  - Hold-out test set evaluation
  - Performance comparison between implementations

### 6. Visualization (`src/utils/visualization.py`)
- **Plot Types**:
  - Scatter plots for actual vs predicted
  - Residual analysis
  - Feature importance visualization
  - Attention weights heatmaps
  - Training history plots
  - Spectral curves

### 7. Utilities
- **Logger** (`src/utils/logger.py`):
  - Structured logging
  - Error tracking
  - Performance monitoring
- **Data Quality** (`src/utils/data_quality.py`):
  - Input validation
  - Data integrity checks
  - Outlier detection
- **Interpretability** (`src/utils/interpretability.py`):
  - Feature importance analysis
  - Model explanation tools
  - Attention mechanism visualization

## Data Flow
1. Raw spectral data → Data Processor → Preprocessed features
2. Preprocessed features → Model Training → Trained model
3. Trained model → Evaluation → Performance metrics
4. Model deployment:
   - Via FastAPI for programmatic access
   - Via Streamlit for user interface
   - Docker containers for scalable deployment

## Configuration
- Model parameters in respective model files
- API settings in environment variables
- Preprocessing parameters in data processor
- Logging configuration in logger utility
- Docker settings in Dockerfile

## Testing
- Unit tests for individual components
- Integration tests for end-to-end workflow
- Performance benchmarks
- CI/CD pipeline validation 