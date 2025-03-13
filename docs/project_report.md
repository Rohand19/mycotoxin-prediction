# DON Concentration Prediction Project Report

## Executive Summary

This report summarizes the development of a machine learning system for predicting Deoxynivalenol (DON) concentration in corn samples using hyperspectral imaging data. The system achieves an R² score of 0.9058, demonstrating strong predictive performance for this agricultural application.

## 1. Data Preprocessing Approach

### 1.1 Data Loading and Inspection

The dataset consists of hyperspectral reflectance values for corn samples, with 448 spectral bands as features and DON concentration as the target variable. Initial data inspection revealed:

- 500 samples with 450 columns (448 spectral bands, 1 target variable, 1 ID column)
- No missing values in the dataset
- Non-numeric ID column that needed special handling
- Varying scales across spectral bands

### 1.2 Preprocessing Pipeline

The preprocessing pipeline implements the following steps:

1. **Data Cleaning**:
   - Separation of non-numeric columns (e.g., sample IDs)
   - Conversion of data to float32 to reduce memory usage
   - Memory optimization using garbage collection

2. **Feature Scaling**:
   - Implementation of RobustScaler for spectral bands to handle potential outliers
   - Use of quantile range (5-95%) to minimize the impact of extreme values

3. **Target Variable Treatment**:
   - Analysis of target distribution skewness
   - Conditional log transformation when skewness exceeds a threshold
   - StandardScaler application to normalize the target variable

4. **Memory Efficiency**:
   - Implementation of memory usage tracking
   - Conversion to memory-efficient data types
   - Strategic garbage collection

### 1.3 Rationale for Preprocessing Choices

- **RobustScaler for Features**: Hyperspectral data often contains outliers due to sensor variations. RobustScaler provides resilience against these outliers by using quantiles instead of mean and standard deviation.

- **Skewness-based Transformation**: DON concentration values can have a skewed distribution. The conditional log transformation helps normalize the distribution when necessary.

- **Memory Optimization**: Hyperspectral datasets can be large. The memory optimization techniques ensure the system can scale to larger datasets.

## 2. Dimensionality Reduction Insights

### 2.1 Approach to Dimensionality Reduction

The model architecture incorporates dimensionality reduction through:

1. **Initial Dense Layer**: Reduction from 448 spectral bands to 256 features
2. **Attention Mechanism**: Further focusing on the most relevant features
3. **Global Average Pooling**: Reducing the dimensionality after attention processing

### 2.2 Key Insights

- **Spectral Redundancy**: Many adjacent spectral bands contain redundant information, allowing for effective dimensionality reduction.

- **Attention Importance**: The attention mechanism successfully identifies the most informative spectral regions for DON prediction, with particular focus on specific wavelength ranges.

- **Feature Importance**: SHAP analysis reveals that certain spectral bands have significantly higher importance for prediction, particularly in the near-infrared region.

## 3. Model Development

### 3.1 Model Selection

After evaluating several approaches, a neural network with a multi-head self-attention mechanism was selected as the final model. This architecture was chosen for its ability to:

- Capture complex non-linear relationships in spectral data
- Focus on the most relevant spectral bands through attention
- Handle the high-dimensional nature of hyperspectral data

### 3.2 Model Architecture

The final model architecture consists of:

1. **Input Layer**: Accepts 448 spectral bands
2. **Batch Normalization**: Stabilizes input data
3. **Dimensionality Reduction**: Dense layer reducing to 256 features
4. **Attention Mechanism**: Multi-head self-attention with 2 heads
5. **Dense Layers**: Multiple dense layers with decreasing units (128, 64, 32)
6. **Regularization**: Batch normalization and dropout layers
7. **Output Layer**: Single unit for DON concentration prediction

### 3.3 Training Approach

- **Data Split**: 80% training, 20% testing
- **Validation Strategy**: 20% of training data used for validation
- **Optimizer**: Adam with learning rate of 0.001
- **Loss Function**: Mean Squared Error
- **Early Stopping**: Based on validation loss with patience
- **Learning Rate Reduction**: On plateau to fine-tune training

### 3.4 Hyperparameter Optimization

Key hyperparameters were optimized:

- L2 regularization strength: 0.001
- Attention heads: 2
- Attention dimension: 16
- Dropout rate: 0.1
- Dense layer configuration: [128, 64, 32]

## 4. Evaluation Results

### 4.1 Performance Metrics

The model achieved excellent performance on the test set:

- **R² Score**: 0.9058 (indicating that the model explains 90.58% of the variance)
- **RMSE**: 5,131.81 ppb
- **MAE**: 3,295.69 ppb

### 4.2 Visual Analysis

- **Actual vs. Predicted Plot**: Shows strong correlation between predicted and actual values
- **Residual Analysis**: Residuals are normally distributed with no systematic patterns
- **Training History**: Loss curves show proper convergence without overfitting

### 4.3 Model Interpretability

SHAP analysis revealed:

- Certain spectral bands have significantly higher importance
- The model focuses on specific wavelength regions associated with mycotoxin presence
- The attention mechanism successfully identifies the most relevant features

## 5. Key Findings and Conclusions

### 5.1 Main Findings

1. Hyperspectral data can effectively predict DON concentration with high accuracy
2. The attention mechanism significantly improves model performance by focusing on relevant spectral regions
3. Robust preprocessing is essential for handling the variability in spectral data
4. Memory optimization techniques enable efficient processing of large hyperspectral datasets

### 5.2 Limitations

1. The model's performance may vary with different corn varieties or growing conditions
2. The current implementation requires all 448 spectral bands for prediction
3. The confidence intervals are simplified and could be improved with more sophisticated uncertainty estimation

### 5.3 Future Improvements

1. **Model Enhancements**:
   - Experiment with different attention mechanisms
   - Implement ensemble methods for improved robustness
   - Explore transfer learning from related spectral tasks

2. **Feature Engineering**:
   - Develop spectral indices specific to DON detection
   - Implement automated feature selection
   - Explore wavelength binning to reduce dimensionality

3. **Deployment Optimizations**:
   - Model quantization for faster inference
   - Implement streaming data processing
   - Develop on-device deployment options for field use

## 6. Conclusion

The developed system demonstrates that deep learning with attention mechanisms can effectively predict DON concentration in corn samples using hyperspectral data. The high R² score of 0.9058 indicates strong predictive performance, making this approach promising for practical agricultural applications.

The modular, production-ready implementation ensures that the system can be easily deployed and integrated into existing agricultural workflows, potentially reducing the need for costly and time-consuming laboratory testing for DON concentration. 