# Streamlit User Guide

## Overview
The DON Concentration Predictor provides two Streamlit web interfaces:
1. **Primary Interface** (`streamlit_app.py`): Uses the TensorFlow-based neural network model
2. **Alternative Interface** (`streamlit_app_simple_sklearn.py`): Uses the RandomForest model

## Running the Apps

### Primary TensorFlow Interface
```bash
streamlit run src/streamlit_app.py
```

### Alternative RandomForest Interface
```bash
streamlit run src/streamlit_app_simple_sklearn.py
```

The app will be available at: `http://localhost:8501`

## Choosing Which App to Use

### TensorFlow App (Primary)
Best for:
- Systems with GPU acceleration
- When maximum accuracy is needed
- Environments with good TensorFlow support
- When you need attention mechanism visualization

### RandomForest App (Alternative)
Best for:
- Apple Silicon Macs
- When TensorFlow issues arise
- When faster predictions are needed
- Systems with limited resources

## Interface Overview

### Common Features (Both Apps)
- Single sample prediction
- Batch prediction support
- Data visualization
- Performance metrics display
- Export predictions to CSV

### TensorFlow-Specific Features
- Attention mechanism visualization
- Confidence intervals for predictions
- Training history plots
- Model architecture display

### RandomForest-Specific Features
- Feature importance plots
- Cross-validation results
- Faster prediction times
- Lower resource usage

## Using the Apps

### Single Prediction
1. Select "Single Prediction" from the sidebar
2. Choose input method:
   - Manual input: Enter 448 spectral values
   - File upload: Upload CSV with spectral data
3. Click "Predict" to get results
4. View prediction and visualizations

### Batch Prediction
1. Select "Batch Prediction" from the sidebar
2. Upload CSV file with multiple samples
3. Click "Predict" to process all samples
4. Download results as CSV

### Data Analysis
1. Select "Data Analysis" from the sidebar
2. Upload your dataset
3. Explore:
   - Spectral curves
   - Data distribution
   - Feature correlations
   - Model performance metrics

## File Formats

### Input Data
- CSV files with 448 columns for spectral data
- Optional sample ID column
- Headers required
- No missing values allowed

Example format:
```csv
sample_id,wavelength_1,wavelength_2,...,wavelength_448
1,0.123,0.456,...,0.789
2,0.234,0.567,...,0.890
```

### Output Data
- CSV files with predictions
- Includes:
  - Original sample IDs
  - Predicted DON concentrations
  - Confidence intervals (TensorFlow only)
  - Prediction timestamps

## Performance Considerations

### TensorFlow App
- Initial loading time: 5-10 seconds
- Single prediction: 1-2 seconds
- Batch prediction: ~5 seconds per 100 samples
- Memory usage: ~500MB

### RandomForest App
- Initial loading time: 2-3 seconds
- Single prediction: <1 second
- Batch prediction: ~2 seconds per 100 samples
- Memory usage: ~100MB

## Troubleshooting

### Common Issues

#### App Crashes
- Check system memory
- Restart the app
- Try the RandomForest version if using TensorFlow

#### File Upload Issues
- Verify CSV format
- Check for missing values
- Ensure correct number of columns (448)

#### Prediction Errors
- Validate input data range
- Check for non-numeric values
- Ensure proper data preprocessing

### TensorFlow-Specific Issues
- GPU memory errors: Switch to CPU mode
- Initialization errors: Check TensorFlow installation
- Memory leaks: Restart app periodically

### RandomForest-Specific Issues
- Memory errors: Reduce batch size
- Slow predictions: Check system resources
- Import errors: Verify scikit-learn installation

## Best Practices

### Data Preparation
- Clean data before upload
- Remove outliers
- Normalize if needed
- Check for missing values

### Batch Processing
- Limit batch size to 1000 samples
- Monitor memory usage
- Save results frequently

### Visualization
- Use appropriate plot types
- Export plots when needed
- Consider data size when plotting

## Getting Help

### Documentation
- README.md: Project overview
- API documentation: For programmatic access
- Technical documentation: Implementation details

### Support
- GitHub issues: Bug reports
- Email support: Technical questions
- Community forum: User discussions

## Updates and Maintenance

### Version Control
- Check version number in app
- Update regularly
- Review changelog

### Data Backup
- Export predictions regularly
- Save visualization results
- Keep input data backups

## Security Considerations

### Data Privacy
- Local processing only
- No data storage
- Secure file handling

### Access Control
- Local network only
- No authentication required
- Firewall configuration recommended

## Future Enhancements
- Real-time prediction updates
- Enhanced visualization options
- Additional model support
- Automated data validation
- Performance optimizations 