# DON Concentration Predictor: Technical Report

## 1. Preprocessing Steps and Rationale

### Data Cleaning
- **Missing Values**: Implemented median imputation for missing spectral values, as it preserves the distribution better than mean imputation for spectral data.
- **Outlier Detection**: Used Isolation Forest for anomaly detection in spectral signatures, with a contamination factor of 0.1.
- **Normalization**: Applied min-max scaling to spectral bands to ensure all features are on the same scale (0-1 range).

### Feature Engineering
- **Spectral Indices**: Created derivative-based spectral indices to capture rate of change in reflectance.
- **Smoothing**: Applied Savitzky-Golay filtering to reduce noise while preserving spectral features.
- **Band Selection**: Used correlation analysis to identify most informative wavelength bands.

## 2. Dimensionality Reduction Insights

### Principal Component Analysis (PCA)
- First 3 components explain 85% of variance
- PC1 strongly correlates with overall reflectance intensity
- PC2 captures spectral slope variations
- PC3 relates to specific absorption features

### t-SNE Analysis
- Revealed distinct clusters in spectral signatures
- Higher DON concentrations tend to cluster together
- Identified potential subgroups within concentration ranges

## 3. Model Selection and Training

### Primary Implementation (TensorFlow)
- **Architecture**: Neural network with attention mechanism
  - Input layer: Spectral bands (normalized)
  - Attention layer: Multi-head self-attention
  - Hidden layers: [256, 128, 64] with ReLU activation
  - Output layer: Single neuron (DON concentration)
- **Performance**:
  - RMSE: 5,131.81 ppb
  - R²: 0.9058
  - MAE: 3,295.69 ppb

### Alternative Implementation (RandomForest)
- **Configuration**: 
  - 100 estimators
  - Max depth: 15
  - Min samples split: 5
- **Performance**:
  - RMSE: 5,487.32 ppb
  - R²: 0.8923
  - MAE: 3,412.45 ppb

## 4. Key Findings

### Model Performance
- Attention mechanism significantly improves prediction accuracy
- RandomForest provides comparable performance with lower computational requirements
- Both models show robust performance across different concentration ranges

### Feature Importance
- SHAP analysis reveals key wavelength regions:
  - 1100-1200nm: Strongest predictive power
  - 1450-1600nm: Secondary importance
  - 900-1000nm: Tertiary importance

### Limitations
- Model performance degrades for extreme DON concentrations
- Requires high-quality spectral data with minimal noise
- Sensitive to sensor calibration differences

## 5. Potential Improvements

### Technical Enhancements
- Implement ensemble methods combining neural network and RandomForest predictions
- Explore transfer learning from pretrained spectral models
- Add uncertainty quantification to predictions

### Production Optimization
- Implement model versioning and A/B testing
- Add automated retraining pipeline
- Enhance monitoring and alerting system

### Data Quality
- Implement automated calibration correction
- Add more robust data quality checks
- Create synthetic data augmentation pipeline 