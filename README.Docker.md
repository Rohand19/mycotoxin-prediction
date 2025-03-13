# Docker Deployment Guide for DON Concentration Predictor

This guide provides instructions for deploying the DON Concentration Predictor using Docker.

## Services

The Docker deployment includes three services:

1. **API Service**: FastAPI-based REST API for predictions
2. **TensorFlow Streamlit App**: Primary web interface using TensorFlow model
3. **RandomForest Streamlit App**: Alternative web interface using RandomForest model

## Building and Running

### Local Development

To build and run all services locally:

```bash
docker compose up --build
```

This will start:
- API service at http://localhost:8000
- TensorFlow Streamlit app at http://localhost:8501
- RandomForest Streamlit app at http://localhost:8502

### Running Individual Services

To run only the API service:

```bash
docker compose up api
```

To run only the TensorFlow Streamlit app:

```bash
docker compose up streamlit
```

To run only the RandomForest Streamlit app:

```bash
docker compose up streamlit_sklearn
```

## Production Deployment

For production deployment, build the image with appropriate platform settings:

```bash
# For same architecture as build machine
docker build -t don-predictor .

# For specific architecture (e.g., deploying from M1 Mac to x86 server)
docker build --platform=linux/amd64 -t don-predictor .
```

Push to your container registry:

```bash
docker tag don-predictor your-registry.com/don-predictor:latest
docker push your-registry.com/don-predictor:latest
```

## Environment Variables

The following environment variables can be set to configure the services:

- `MODEL_PATH`: Path to the model file (default: models/best_model.keras)
- `X_SCALER_PATH`: Path to the X scaler file (default: models/X_scaler.pkl)
- `Y_SCALER_PATH`: Path to the Y scaler file (default: models/y_scaler.pkl)
- `LOG_LEVEL`: Logging level (default: INFO)

## Health Checks

The API service includes a health check endpoint at `/health` that returns the service status and memory usage.

## Troubleshooting

If you encounter issues with the TensorFlow model on certain platforms (e.g., Apple Silicon Macs), try using the RandomForest implementation instead:

```bash
docker compose up streamlit_sklearn
```

## References
* [Docker's Python guide](https://docs.docker.com/language/python/)
* [FastAPI deployment](https://fastapi.tiangolo.com/deployment/)
* [Streamlit deployment](https://docs.streamlit.io/knowledge-base/deploy/)