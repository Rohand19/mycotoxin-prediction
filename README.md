# Corn DON Concentration Predictor

This project implements a machine learning pipeline for predicting DON (Deoxynivalenol) concentration in corn samples using hyperspectral imaging data. The system uses a deep learning model with attention mechanism to achieve high accuracy in predictions.

## Performance Metrics

The model achieves the following performance on the test set:
- R² Score: 0.9058
- RMSE: 5,131.81 ppb
- MAE: 3,295.69 ppb

## Project Structure

```
.
├── app.py                  # FastAPI application
├── attention.py           # Attention mechanism implementation
├── data_preprocessing.py  # Data preprocessing functions
├── Dockerfile            # Docker configuration
├── main.py              # Main training script
├── model.py             # Model architecture
├── requirements.txt     # Project dependencies
├── streamlit_app.py     # Streamlit web interface
├── visualization.py     # Visualization utilities
└── tests/              # Unit tests
```

## Features

- Data preprocessing pipeline with automated quality checks
- Deep learning model with attention mechanism
- Comprehensive visualization tools
- FastAPI REST API for predictions
- Interactive Streamlit web interface
- Docker containerization
- Unit tests and documentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/corn-don-predictor.git
cd corn-don-predictor
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model:
```bash
python main.py
```

### Running the API

Start the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Running the Streamlit App

Launch the web interface:
```bash
streamlit run streamlit_app.py
```

### Docker Deployment

Build and run the Docker container:
```bash
docker build -t corn-don-predictor .
docker run -p 8000:8000 corn-don-predictor
```

## API Documentation

The API provides the following endpoints:

- `GET /`: Root endpoint with API information
- `POST /predict`: Make predictions from spectral data
- `GET /health`: Health check endpoint

Example API request:
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "features": [/* 448 spectral values */]
}
response = requests.post(url, json=data)
prediction = response.json()
```

## Model Architecture

The model uses a combination of:
- Multi-head self-attention layers
- Batch normalization
- Dropout for regularization
- Dense layers with L2 regularization

## Data Preprocessing

The pipeline includes:
- Missing value handling
- Feature scaling
- Outlier detection
- Automated quality checks

## Visualization Tools

The project includes tools for:
- Spectral data visualization
- Feature correlation analysis
- Training history plots
- Prediction results visualization
- Data quality assessment

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the TensorFlow team for their excellent framework
- The attention mechanism implementation is inspired by the Transformer architecture 