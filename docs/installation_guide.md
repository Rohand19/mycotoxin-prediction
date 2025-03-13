# Installation Guide

This guide provides detailed instructions for setting up the DON Concentration Predictor project on your system.

## Prerequisites

Before installing the project, ensure you have the following prerequisites:

- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- 4GB+ RAM (recommended for model training)
- 2GB+ disk space

## Installation Options

There are three ways to install and run the project:

1. **Standard Installation**: Install directly on your system
2. **Virtual Environment**: Install in an isolated Python environment (recommended)
3. **Docker Installation**: Run in a Docker container

## Option 1: Standard Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/don-concentration-predictor.git
cd don-concentration-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python run_tests.py
```

If all tests pass, the installation was successful.

## Option 2: Virtual Environment (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/don-concentration-predictor.git
cd don-concentration-predictor
```

### 2. Create a Virtual Environment

#### On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python run_tests.py
```

If all tests pass, the installation was successful.

### 5. Deactivate the Virtual Environment (When Done)

```bash
deactivate
```

## Option 3: Docker Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/don-concentration-predictor.git
cd don-concentration-predictor
```

### 2. Build the Docker Image

```bash
docker build -t don-predictor .
```

### 3. Run the Docker Container

#### For API Service:

```bash
docker run -p 8000:8000 don-predictor
```

#### For Streamlit App:

```bash
docker run -p 8501:8501 don-predictor streamlit
```

#### For Both Services:

```bash
docker run -p 8000:8000 -p 8501:8501 don-predictor both
```

## Troubleshooting Installation Issues

### Common Issues

#### 1. Package Installation Errors

If you encounter errors during package installation:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If specific packages fail to install, you can try installing them individually:

```bash
pip install numpy pandas scikit-learn tensorflow
pip install fastapi uvicorn streamlit
```

#### 2. TensorFlow Installation Issues

On some systems, TensorFlow may require additional setup:

##### For macOS:

```bash
pip install tensorflow-macos
```

##### For systems without GPU:

```bash
pip install tensorflow-cpu
```

#### 3. Permission Issues

If you encounter permission errors:

##### On Linux/macOS:

```bash
sudo pip install -r requirements.txt
```

##### On Windows:

Run the command prompt as administrator.

#### 4. Virtual Environment Issues

If you have problems with the virtual environment:

```bash
python -m pip install --upgrade virtualenv
python -m virtualenv venv
```

Then activate the environment as described above.

## System-Specific Instructions

### macOS

For macOS users, you might need to install additional dependencies:

```bash
brew install libomp
```

### Windows

For Windows users, ensure you have the Microsoft Visual C++ Redistributable installed, which is required for some Python packages.

### Linux

For Linux users, you might need to install additional system packages:

```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libopenblas-dev
```

## Development Setup

If you plan to contribute to the project, you'll need additional tools:

### 1. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 3. Install Testing Tools

```bash
pip install pytest pytest-cov
```

## Running the Project

After installation, you can run the project components:

### 1. Train the Model

```bash
python src/train.py
```

### 2. Run the API Service

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Run the Streamlit App

```bash
streamlit run src/streamlit_app.py
```

## Next Steps

After installation, refer to the following documentation:

- [Project Report](project_report.md): Overview of the project approach and findings
- [Technical Documentation](technical_documentation.md): Detailed technical information
- [API Documentation](api_documentation.md): Information about the API endpoints
- [Streamlit User Guide](streamlit_user_guide.md): Guide for using the web interface 