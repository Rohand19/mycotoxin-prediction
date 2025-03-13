# DON Concentration Predictor Documentation

Welcome to the documentation for the DON Concentration Predictor system. This documentation provides comprehensive information about the system's architecture, installation, usage, and technical details.

## System Overview

The DON Concentration Predictor is a machine learning system designed to predict Deoxynivalenol (DON) concentration in corn samples using hyperspectral imaging data. The system processes spectral reflectance data to make accurate predictions of mycotoxin levels.

### Implementation Approaches

The system offers two implementation approaches:

1. **Primary Implementation (TensorFlow)**: A neural network with multi-head self-attention mechanism, providing high accuracy and sophisticated feature analysis.

2. **Alternative Implementation (scikit-learn)**: A RandomForest regressor offering comparable performance with lower computational requirements, particularly useful for environments where TensorFlow may not be optimal (e.g., Apple Silicon Macs).

## Key Features

- **Advanced Data Processing**: Robust preprocessing pipeline with automatic handling of non-numeric data, outliers, and skewness.

- **Dual Model Architecture**:
  - **TensorFlow Neural Network**: With attention mechanism for capturing complex spectral relationships
  - **scikit-learn RandomForest**: For environments with limited TensorFlow support

- **Comprehensive Evaluation**: Multiple metrics including MAE, RMSE, R², and residual analysis.

- **Interactive Interfaces**: 
  - FastAPI service for programmatic access
  - Streamlit web interface for interactive use (both TensorFlow and RandomForest versions)

- **Production-Ready**: Docker containerization, comprehensive logging, and memory usage tracking.

## Documentation Contents

### User Guides

- [Installation Guide](installation_guide.md): Instructions for setting up the system in various environments.
- [Streamlit User Guide](streamlit_user_guide.md): Guide to using the Streamlit web interface.

### Technical Documentation

- [Technical Documentation](technical_documentation.md): Detailed information about the system architecture and components.
- [API Documentation](api_documentation.md): Documentation for the FastAPI service.
- [Project Report](project_report.md): Overview of the project approach and findings.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/don-concentration-predictor.git
cd don-concentration-predictor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Interface

For the primary TensorFlow-based interface:

```bash
streamlit run src/streamlit_app.py
```

For the alternative RandomForest-based interface (recommended for Apple Silicon Macs):

```bash
streamlit run src/streamlit_app_simple_sklearn.py
```

### Starting the API Service

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Choosing Between Implementations

- **Use the TensorFlow implementation when**:
  - You have GPU acceleration available
  - Maximum accuracy is required
  - You're working on a system with good TensorFlow support

- **Use the RandomForest implementation when**:
  - You're working on Apple Silicon Macs or systems with limited TensorFlow support
  - You need faster inference with comparable accuracy
  - You prefer a simpler model with built-in feature importance

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU recommended for TensorFlow implementation (but not required)
- For Apple Silicon Macs: Consider using the RandomForest implementation for optimal performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact [your.email@example.com](mailto:your.email@example.com). 