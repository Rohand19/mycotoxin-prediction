# API Documentation

The DON Concentration Predictor provides a RESTful API built with FastAPI for making predictions programmatically. This document describes the available endpoints, request/response formats, and usage examples.

## API Overview

The API service is designed to provide DON concentration predictions based on hyperspectral data. It can be configured to use either the primary TensorFlow model or the alternative RandomForest model, depending on your deployment environment and requirements.

### Base URL

When running locally, the API is available at:

```
http://localhost:8000
```

## Endpoints

### Root Endpoint

```
GET /
```

Returns basic information about the API.

#### Response

```json
{
  "message": "DON Concentration Prediction API",
  "version": "1.0.0",
  "docs_url": "/docs",
  "redoc_url": "/redoc"
}
```

### Health Check

```
GET /health
```

Returns the current status of the API service, including whether the model is loaded and the current memory usage.

#### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory_usage": 1234.56
}
```

### Predict DON Concentration

```
POST /predict
```

Makes a prediction of DON concentration based on input spectral data.

#### Request Body

The request body should be a JSON object with a `values` field containing an array of 448 spectral reflectance values.

```json
{
  "values": [0.1, 0.2, 0.3, ..., 0.5]  // 448 values required
}
```

#### Response

The response is a JSON object containing the predicted DON concentration in ppb (parts per billion).

When using the TensorFlow model (primary implementation):

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

When using the RandomForest model (alternative implementation):

```json
{
  "don_concentration": 123.45,
  "units": "ppb"
}
```

#### Error Responses

If the input data is invalid or an error occurs during prediction, the API will return an appropriate error response:

```json
{
  "detail": "Error message describing the issue"
}
```

Common error scenarios include:
- Invalid number of spectral bands (not 448)
- Non-numeric values in the input data
- Internal server errors during prediction

## API Configuration

The API can be configured to use either the TensorFlow model (primary) or the RandomForest model (alternative) by modifying the `src/api/main.py` file. By default, it uses the TensorFlow model.

### Using the TensorFlow Model (Primary)

The TensorFlow model provides higher accuracy and confidence intervals but requires TensorFlow dependencies and may not work optimally on all platforms (e.g., Apple Silicon Macs).

### Using the RandomForest Model (Alternative)

The RandomForest model provides comparable accuracy without TensorFlow dependencies, making it more suitable for environments where TensorFlow may not be optimal.

## Usage Examples

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"values": [0.1, 0.2, ..., 0.5]}'  # 448 values
```

### Python

```python
import requests
import numpy as np

# Generate sample data (448 spectral bands)
values = np.random.rand(448).tolist()

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={"values": values}
)

# Print result
if response.status_code == 200:
    result = response.json()
    print(f"Predicted DON concentration: {result['don_concentration']} {result['units']}")
else:
    print(f"Error: {response.json()['detail']}")
```

## Interactive Documentation

FastAPI provides interactive documentation for the API. When the API is running, you can access:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These interfaces allow you to explore the API endpoints and make test requests directly from your browser.

## Running the API

To start the API service:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn src.api.main:app --reload
```

## Docker Deployment

The API can be deployed using Docker:

```bash
# Build the Docker image
docker build -t don-predictor .

# Run the container
docker run -p 8000:8000 don-predictor
```

## Error Handling

The API implements comprehensive error handling to provide informative error messages. Common errors include:

1. **Validation Errors**: When the input data doesn't match the expected format
2. **Model Errors**: When the model encounters an issue during prediction
3. **Server Errors**: When an unexpected error occurs in the API service

Each error response includes a descriptive message to help diagnose and resolve the issue.

## Performance Considerations

- The API is designed to handle concurrent requests efficiently
- Prediction time may vary depending on the model used (TensorFlow vs. RandomForest)
- For high-throughput scenarios, consider using the RandomForest model which typically has faster inference times
- Memory usage is monitored and can be checked via the `/health` endpoint 