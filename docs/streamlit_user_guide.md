# Streamlit User Guide

This guide provides instructions for using the DON Concentration Predictor's Streamlit web interface. The system offers two Streamlit applications:

1. **Primary Interface** (`streamlit_app.py`): Uses the TensorFlow-based neural network model
2. **Alternative Interface** (`streamlit_app_simple_sklearn.py`): Uses the RandomForest model, particularly useful for Apple Silicon Macs

## Getting Started

### Running the Streamlit App

To run the primary TensorFlow-based Streamlit app:

```bash
streamlit run src/streamlit_app.py
```

To run the alternative RandomForest-based Streamlit app:

```bash
streamlit run src/streamlit_app_simple_sklearn.py
```

The app will start and automatically open in your default web browser. If it doesn't open automatically, you can access it at:

```
http://localhost:8501
```

### Choosing Which App to Use

- **Use the TensorFlow-based app** (`streamlit_app.py`) when:
  - You have GPU acceleration available
  - You're working on a system with good TensorFlow support
  - You need maximum accuracy and confidence intervals

- **Use the RandomForest-based app** (`streamlit_app_simple_sklearn.py`) when:
  - You're working on Apple Silicon Macs
  - You encounter TensorFlow-related issues
  - You need faster predictions with comparable accuracy

## Interface Overview

Both Streamlit apps provide a user-friendly interface for predicting DON concentration in corn samples using hyperspectral data. The interface is divided into several sections:

### Navigation

The sidebar navigation allows you to access different features of the app:

- **Single Prediction**: Make predictions for a single sample
- **Batch Prediction**: Make predictions for multiple samples
- **Data Analysis**: Analyze and visualize data quality

## Single Prediction

The Single Prediction page allows you to predict DON concentration for a single corn sample.

### Input Methods

There are two ways to input spectral data:

1. **Manual Input**: Enter spectral values directly in the interface
2. **File Upload**: Upload a CSV file containing spectral data

#### Manual Input

For manual input:

1. Select the "Manual Input" option
2. Enter spectral values for each band (the interface shows input fields for the first 10 bands)
3. Click the "Predict" button

Note: While the interface only shows input fields for the first 10 bands to keep the UI manageable, the model uses all 448 bands. Default values are used for bands not explicitly specified.

#### File Upload (Single Sample)

For file upload:

1. Select the "File Upload" option
2. Upload a CSV file containing spectral data for a single sample
   - The file should have 448 columns representing spectral bands
   - The file should have a header row
3. Click the "Predict" button

### Prediction Results

After making a prediction, the app displays:

- Predicted DON concentration in ppb
- Confidence interval (TensorFlow model only)
- Visualization of the spectral curve

## Batch Prediction

The Batch Prediction page allows you to predict DON concentration for multiple corn samples at once.

### File Upload (Multiple Samples)

To make batch predictions:

1. Upload a CSV file containing spectral data for multiple samples
   - Each row should represent a sample
   - The file should have 448 columns representing spectral bands
   - The file should have a header row
2. Click the "Predict" button

### Batch Results

After making batch predictions, the app displays:

- Table of predictions for each sample
- Summary statistics (mean, median, min, max)
- Histogram of predicted DON concentrations
- Option to download results as CSV

## Data Analysis

The Data Analysis page provides tools for analyzing and visualizing hyperspectral data.

### Data Upload

To analyze data:

1. Upload a CSV file containing spectral data
   - The file should have samples as rows and spectral bands as columns
   - The file should have a header row
2. Click the "Analyze" button

### Analysis Features

The data analysis includes:

- Data quality summary (missing values, outliers)
- Spectral curve visualization
- Principal Component Analysis (PCA) visualization
- Correlation heatmap
- Feature importance analysis (method varies by implementation)

## Implementation Differences

While both Streamlit apps provide similar functionality, there are some differences between the TensorFlow and RandomForest implementations:

### TensorFlow Implementation (`streamlit_app.py`)

- Uses a neural network with attention mechanism
- Provides confidence intervals for predictions
- Uses SHAP values for feature importance
- May be slower on systems without GPU acceleration
- May encounter issues on Apple Silicon Macs

### RandomForest Implementation (`streamlit_app_simple_sklearn.py`)

- Uses a RandomForest regressor
- Does not provide confidence intervals
- Uses built-in feature importance from RandomForest
- Generally faster for predictions
- More reliable on Apple Silicon Macs
- More robust to missing or invalid data

## Troubleshooting

### Common Issues

#### App Freezes or Crashes

If the app freezes or crashes:

- For TensorFlow app: Try using the RandomForest app instead
- Restart the app with `streamlit run src/streamlit_app_simple_sklearn.py`
- Check system memory usage (the app may require significant memory)

#### File Upload Issues

If you encounter issues with file uploads:

- Ensure your CSV file has the correct format (448 columns for spectral bands)
- Check for non-numeric values in your data
- Try with a smaller file first

#### Prediction Errors

If you receive prediction errors:

- Check that your input data has the correct number of spectral bands (448)
- Ensure all values are numeric
- Verify that the model files are present in the `models` directory

## Advanced Usage

### Customizing the App

You can customize the app by modifying the source code:

- `src/streamlit_app.py`: Primary TensorFlow-based app
- `src/streamlit_app_simple_sklearn.py`: Alternative RandomForest-based app

### Running with Different Models

The apps are configured to use specific models by default:

- TensorFlow app uses `models/best_model.keras`
- RandomForest app uses `models/rf_model_real_data.joblib`

To use different models, you can modify the model loading code in the respective app files.

## Feedback and Support

If you encounter issues or have suggestions for improving the Streamlit interface, please:

1. Check the documentation for solutions
2. Submit an issue on the GitHub repository
3. Contact the project maintainers 