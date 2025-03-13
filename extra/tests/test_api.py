import pytest
from fastapi.testclient import TestClient
from app import app
import numpy as np

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "DON Prediction API"
    assert response.json()["version"] == "1.0.0"
    assert response.json()["status"] == "active"

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid input."""
    # Create valid input data
    features = np.random.normal(size=448).tolist()
    response = client.post(
        "/predict",
        json={"features": features}
    )
    
    assert response.status_code == 200
    assert "don_concentration" in response.json()
    assert "confidence_interval" in response.json()
    assert "lower_bound" in response.json()["confidence_interval"]
    assert "upper_bound" in response.json()["confidence_interval"]

def test_predict_endpoint_invalid_input():
    """Test prediction endpoint with invalid input."""
    # Test with wrong number of features
    features = np.random.normal(size=10).tolist()
    response = client.post(
        "/predict",
        json={"features": features}
    )
    
    assert response.status_code == 400
    assert "Expected 448 features" in response.json()["detail"]

def test_predict_endpoint_invalid_json():
    """Test prediction endpoint with invalid JSON."""
    response = client.post(
        "/predict",
        json={"wrong_key": [1, 2, 3]}
    )
    
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_missing_data():
    """Test prediction endpoint with missing data."""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error 