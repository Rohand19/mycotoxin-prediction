# API Documentation

## Overview
The DON Concentration Predictor API provides a RESTful interface for making predictions using either the TensorFlow-based neural network model or the RandomForest model. The API is built with FastAPI for high performance and includes comprehensive error handling and validation.

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint
```
GET /
```
Returns basic information about the API.

**Response**:
```json
{
    "name": "DON Concentration Predictor API",
    "version": "1.0.0",
    "description": "API for predicting DON concentration in corn samples",
    "model_type": "tensorflow"  // or "randomforest"
}
```

### 2. Health Check
```
GET /health
```
Returns the health status of the API.

**Response**:
```json
{
    "status": "healthy",
    "memory_usage": "128MB",
    "model_loaded": true,
    "model_type": "tensorflow"  // or "randomforest"
}
```

### 3. Prediction Endpoint
```
POST /predict
```
Makes predictions for DON concentration based on hyperspectral data.

**Request Body**:
```json
{
    "spectral_data": [
        [0.123, 0.456, ...],  // 448 wavelength values
        [0.789, 0.321, ...]   // For batch predictions
    ]
}
```

**Response**:
```json
{
    "predictions": [
        {
            "don_concentration": 1234.56,
            "confidence_interval": {
                "lower": 1100.0,
                "upper": 1300.0
            }
        },
        {
            "don_concentration": 5678.90,
            "confidence_interval": {
                "lower": 5500.0,
                "upper": 5800.0
            }
        }
    ],
    "model_type": "tensorflow",  // or "randomforest"
    "processing_time": 0.123     // seconds
}
```

## Error Handling

### Error Response Format
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Detailed error message",
        "details": {
            "additional": "information"
        }
    }
}
```

### Common Error Codes
- `INVALID_INPUT`: Input data format is incorrect
- `MISSING_DATA`: Required data is missing
- `MODEL_ERROR`: Error during model prediction
- `VALIDATION_ERROR`: Input validation failed
- `SERVER_ERROR`: Internal server error

## Usage Examples

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"spectral_data": [[0.123, 0.456, ...]]}'
```

### Python
```python
import requests
import numpy as np

# Prepare spectral data
spectral_data = np.random.rand(1, 448).tolist()

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"spectral_data": spectral_data}
)

# Print results
print(response.json())
```

## Running the API

### Local Development
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Docker
```bash
# Build image
docker build -t don-predictor-api .

# Run container
docker run -p 8000:8000 don-predictor-api
```

## Configuration

### Environment Variables
- `MODEL_TYPE`: Choose between "tensorflow" or "randomforest" (default: "tensorflow")
- `MODEL_PATH`: Path to the model file
- `SCALER_X_PATH`: Path to the feature scaler
- `SCALER_Y_PATH`: Path to the target scaler
- `LOG_LEVEL`: Logging level (default: "INFO")
- `MAX_BATCH_SIZE`: Maximum allowed batch size (default: 100)

### Model Selection
The API can be configured to use either the TensorFlow or RandomForest model:

1. **TensorFlow Model**:
   - Higher accuracy with complex patterns
   - Better performance on large datasets
   - Includes attention mechanism visualization
   - Requires more computational resources

2. **RandomForest Model**:
   - Faster inference time
   - Better performance on Apple Silicon
   - Lower resource requirements
   - Built-in feature importance

## Performance Considerations

### Batch Processing
- Maximum batch size is configurable via `MAX_BATCH_SIZE`
- Larger batches improve throughput but increase response time
- Recommended batch size: 32-64 samples

### Memory Usage
- TensorFlow model: ~500MB
- RandomForest model: ~100MB
- Additional ~50MB for API service

### Response Times
- Single prediction: 50-100ms
- Batch prediction (32 samples): 200-300ms
- Response times may vary based on hardware

## Security

### Input Validation
- Validates spectral data shape and values
- Checks for missing or invalid data
- Enforces maximum batch size limits

### Rate Limiting
- Default: 100 requests per minute per IP
- Configurable via environment variables
- Returns 429 status code when exceeded

### Error Handling
- Sanitizes error messages
- Logs detailed errors internally
- Returns safe error messages to clients

## Monitoring

### Metrics
- Request count and latency
- Memory usage
- Model prediction time
- Error rates

### Health Checks
- Model availability
- Memory usage
- System resources
- Dependencies status

## API Versioning
Current version: 1.0.0
- Major version: Breaking changes
- Minor version: New features
- Patch version: Bug fixes 