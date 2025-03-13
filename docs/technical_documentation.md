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

## Component Details

### 1. Data Processing Module

**Location**: `src/preprocessing/data_processor.py`

**Purpose**: Handles data loading, cleaning, and preprocessing for the DON prediction model.

**Key Classes and Methods**:

- `DataProcessor`: Main class for data preprocessing
  - `load_data(file_path)`: Loads data from CSV file
  - `preprocess(df)`: Performs preprocessing steps on the data
  - `inverse_transform_target(y_scaled)`: Converts scaled target values back to original scale
  - `save_scalers(x_scaler_path, y_scaler_path)`: Saves fitted scalers to disk
  - `load_scalers(x_scaler_path, y_scaler_path)`: Loads saved scalers from disk

**Implementation Details**:

- Uses `RobustScaler` for feature scaling to handle outliers
- Implements skewness analysis for target variable transformation
- Handles non-numeric columns automatically
- Optimizes memory usage with garbage collection and efficient data types

### 2. Primary Model Architecture (TensorFlow)

**Location**: `src/models/don_predictor.py`, `src/models/attention.py`

**Purpose**: Defines the neural network architecture for DON concentration prediction.

**Key Classes and Methods**:

- `DONPredictor`: Main model class
  - `build()`: Constructs the model architecture
  - `save(filepath)`: Saves the model to disk
  - `load(filepath)`: Loads a saved model
  - `summary()`: Prints model summary

- `MultiHeadSelfAttention`: Attention mechanism implementation
  - `build(input_shape)`: Builds the attention layers
  - `call(inputs)`: Forward pass through the attention mechanism

**Implementation Details**:

- Neural network with multi-head self-attention
- Dimensionality reduction from 448 to 256 features
- Multiple dense layers with batch normalization and dropout
- L2 regularization to prevent overfitting
- Adam optimizer with learning rate reduction on plateau

### 3. Alternative Model Architecture (scikit-learn)

**Location**: `src/models/simple_predictor.py`

**Purpose**: Provides an alternative implementation using RandomForest for environments where TensorFlow may not be optimal.

**Key Classes and Methods**:

- `SimplePredictor`: Main class for the RandomForest implementation
  - `create_model()`: Creates a new RandomForest model
  - `fit(X, y)`: Trains the model on provided data
  - `predict(X)`: Makes predictions with the model
  - `save(filepath)`: Saves the model to disk
  - `load_or_create(filepath)`: Loads a saved model or creates a new one
  - `save_scalers(x_scaler_path, y_scaler_path)`: Saves fitted scalers to disk

**Implementation Details**:

- Uses scikit-learn's RandomForest regressor
- Includes internal scaling functionality
- Handles feature mismatch automatically
- Provides fallback mechanisms for robustness
- Particularly useful for Apple Silicon Macs and environments without GPU support

### 4. Training Module

**Location**: `src/models/trainer.py`, `src/train.py`

**Purpose**: Handles model training, validation, and checkpointing.

**Key Classes and Methods**:

- `ModelTrainer`: Training orchestration class
  - `train(X_train, y_train, validation_data)`: Trains the model
  - `_setup_callbacks()`: Configures training callbacks
  - `_monitor_memory_usage()`: Tracks memory usage during training

**Implementation Details**:

- Implements early stopping to prevent overfitting
- Uses learning rate reduction on plateau
- Saves best model based on validation loss
- Tracks and logs memory usage during training
- Implements training/validation split

### 5. Evaluation Module

**Location**: `src/utils/metrics.py`

**Purpose**: Provides metrics calculation and evaluation tools.

**Key Functions**:

- `calculate_metrics(y_true, y_pred)`: Calculates regression metrics
- `calculate_confidence_intervals(y_pred, confidence=0.95)`: Estimates prediction confidence intervals
- `calculate_residuals(y_true, y_pred)`: Computes prediction residuals

**Implementation Details**:

- Calculates MAE, RMSE, R² Score
- Computes mean and standard deviation of residuals
- Identifies maximum prediction error

### 6. Visualization Module

**Location**: `src/utils/visualization.py`

**Purpose**: Generates visualizations for data analysis and model evaluation.

**Key Functions**:

- `plot_training_history(history)`: Visualizes training and validation metrics
- `plot_predictions(y_true, y_pred)`: Creates scatter plot of actual vs. predicted values
- `plot_residuals(residuals)`: Visualizes prediction residuals
- `plot_feature_importance(importance_values, feature_names)`: Displays feature importance

**Implementation Details**:

- Uses Matplotlib and Seaborn for visualization
- Saves plots to the visualizations directory
- Implements consistent styling across visualizations

### 7. API Service

**Location**: `src/api/main.py`

**Purpose**: Provides a RESTful API for model predictions.

**Key Endpoints**:

- `GET /`: Root endpoint with API information
- `POST /predict`: Makes DON concentration predictions
- `GET /health`: Health check with memory usage information

**Implementation Details**:

- Built with FastAPI for high performance
- Includes request validation with Pydantic models
- Provides confidence intervals for predictions
- Implements comprehensive error handling
- Includes health monitoring endpoint
- Can be configured to use either the TensorFlow or RandomForest model

### 8. Web Interface

**Location**: `src/streamlit_app.py` (Primary TensorFlow version), `src/streamlit_app_simple_sklearn.py` (Alternative RandomForest version)

**Purpose**: Provides a user-friendly web interface for model interaction.

**Key Features**:

- Data upload and visualization
- Real-time predictions
- Model performance metrics display
- Feature importance visualization

**Implementation Details**:

- Built with Streamlit for interactive UI
- Includes file upload functionality
- Displays visualizations of predictions and model performance
- Shows feature importance (SHAP for TensorFlow, built-in importance for RandomForest)
- Two versions available for different environments and use cases

## Data Flow

1. **Data Ingestion**:
   - Raw hyperspectral data is loaded from CSV files
   - Non-numeric columns are identified and handled

2. **Preprocessing**:
   - Features are scaled using RobustScaler
   - Target variable is analyzed for skewness and transformed if needed
   - Data is split into training and testing sets

3. **Model Training**:
   - Primary: Neural network with attention is constructed
   - Alternative: RandomForest regressor is initialized
   - Model is trained on the training data
   - Best model is saved based on validation performance

4. **Prediction**:
   - New data is preprocessed using the same pipeline
   - Model makes predictions on the processed data
   - Predictions are inverse-transformed to original scale
   - Confidence intervals are calculated

5. **Deployment**:
   - Model and scalers are loaded by the API service
   - API accepts spectral data and returns predictions
   - Streamlit app provides a user interface for the model

## Configuration

The system uses configuration parameters in several components:

1. **Data Processor Configuration**:
   - `robust_quantile_range`: Range for RobustScaler (default: 5.0, 95.0)
   - `skewness_threshold`: Threshold for log transformation (default: 1.0)
   - `target_column`: Name of the target column (default: 'vomitoxin_ppb')
   - `id_column`: Name of the ID column (default: 'hsi_id')

2. **TensorFlow Model Configuration**:
   - `l2_lambda`: L2 regularization strength (default: 0.001)
   - `attention_heads`: Number of attention heads (default: 2)
   - `attention_dim`: Dimension of attention heads (default: 16)
   - `dropout_rate`: Dropout rate (default: 0.1)
   - `dense_layers`: Configuration of dense layers (default: [128, 64, 32])
   - `learning_rate`: Initial learning rate (default: 0.001)

3. **RandomForest Model Configuration**:
   - `n_estimators`: Number of trees in the forest (default: 100)
   - `random_state`: Random seed for reproducibility (default: 42)
   - `n_jobs`: Number of parallel jobs (default: -1, using all cores)

4. **Training Configuration**:
   - `batch_size`: Batch size for training (default: 32)
   - `epochs`: Maximum number of epochs (default: 100)
   - `validation_split`: Fraction of data for validation (default: 0.2)
   - `early_stopping_patience`: Patience for early stopping (default: 10)
   - `reduce_lr_patience`: Patience for learning rate reduction (default: 5)

## Error Handling

The system implements comprehensive error handling:

1. **Data Processing Errors**:
   - Handles missing files
   - Detects and reports non-numeric data
   - Validates data shapes and types

2. **Model Errors**:
   - Validates input shapes
   - Handles training failures
   - Provides informative error messages

3. **API Errors**:
   - Validates request data
   - Returns appropriate HTTP status codes
   - Includes detailed error messages

## Logging

The system uses Python's logging module for comprehensive logging:

1. **Log Levels**:
   - INFO: General information about processing steps
   - WARNING: Potential issues that don't stop execution
   - ERROR: Errors that prevent successful operation

## Choosing Between Implementations

The system provides two implementations to accommodate different environments and use cases:

1. **Primary TensorFlow Implementation**:
   - **Advantages**:
     - Higher accuracy with complex patterns
     - Better performance on large datasets
     - Attention mechanism for feature importance
   - **Best for**:
     - Systems with GPU acceleration
     - When maximum accuracy is required
     - Environments with good TensorFlow support

2. **Alternative RandomForest Implementation**:
   - **Advantages**:
     - More robust on Apple Silicon Macs
     - Faster training and inference
     - Built-in feature importance
     - No TensorFlow dependencies
   - **Best for**:
     - Apple Silicon Macs
     - Systems with limited resources
     - When TensorFlow is problematic
     - When faster inference is needed

The choice between implementations depends on the specific requirements and constraints of the deployment environment. 