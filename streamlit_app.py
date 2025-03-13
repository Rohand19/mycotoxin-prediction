import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import DataVisualizer

# Page config
st.set_page_config(
    page_title="DON Concentration Predictor",
    page_icon="🌽",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model and scalers."""
    model = tf.keras.models.load_model('don_prediction_model.keras')
    X_scaler = joblib.load('X_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    return model, X_scaler, y_scaler

def predict_don(data, model, X_scaler, y_scaler):
    """Make predictions using the loaded model."""
    # Scale features
    data_scaled = X_scaler.transform(data)
    
    # Make prediction
    pred_scaled = model.predict(data_scaled)
    
    # Convert back to original scale
    prediction = y_scaler.inverse_transform(pred_scaled)
    
    return prediction

def main():
    st.title("🌽 Corn DON Concentration Predictor")
    st.write("""
    This application predicts DON (Deoxynivalenol) concentration in corn samples 
    using hyperspectral imaging data.
    """)
    
    # Load model and scalers
    try:
        model, X_scaler, y_scaler = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File upload
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your hyperspectral data (CSV format)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            data = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Basic data statistics
            st.subheader("Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Shape:", data.shape)
                st.write("Missing values:", data.isnull().sum().sum())
            
            with col2:
                st.write("Features:", data.shape[1])
                st.write("Samples:", data.shape[0])
            
            # Make predictions
            if st.button("Make Predictions"):
                with st.spinner("Making predictions..."):
                    predictions = predict_don(data, model, X_scaler, y_scaler)
                
                # Display results
                st.subheader("Prediction Results")
                results_df = pd.DataFrame({
                    'Sample': range(1, len(predictions) + 1),
                    'Predicted DON (ppb)': predictions.flatten()
                })
                st.dataframe(results_df)
                
                # Plot predictions distribution
                st.subheader("Predictions Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(predictions.flatten(), kde=True)
                plt.title("Distribution of Predicted DON Concentrations")
                plt.xlabel("DON Concentration (ppb)")
                plt.ylabel("Count")
                st.pyplot(fig)
                
                # Download predictions
                st.download_button(
                    label="Download Predictions",
                    data=results_df.to_csv(index=False),
                    file_name="don_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    
    # Additional information
    st.sidebar.header("About")
    st.sidebar.write("""
    This application uses a deep learning model with attention mechanism 
    to predict DON concentration in corn samples using hyperspectral data.
    
    The model achieves:
    - R² Score: 0.9058
    - RMSE: 5131.81 ppb
    - MAE: 3295.69 ppb
    """)
    
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Upload a CSV file containing hyperspectral data
    2. Each row should represent one sample
    3. The data should contain 448 spectral features
    4. Click 'Make Predictions' to get results
    """)

if __name__ == "__main__":
    main() 