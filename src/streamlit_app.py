"""Streamlit app for DON concentration prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from models.don_predictor import DONPredictor
from preprocessing.data_processor import DataProcessor
from utils.data_quality import DataQualityAnalyzer
from utils.metrics import calculate_metrics
from utils.visualization import plot_predictions

def load_model_and_processor():
    """Load the trained model and data processor."""
    try:
        model = DONPredictor.load('models/best_model.keras')
        processor = DataProcessor()
        processor.load_scalers('models/X_scaler.pkl', 'models/y_scaler.pkl')
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def main():
    """Main function for the Streamlit app."""
    st.title("DON Concentration Predictor")
    st.write("""
    This application predicts DON (Deoxynivalenol) concentration in corn samples 
    using hyperspectral imaging data. Upload your data or use the interactive 
    prediction feature.
    """)
    
    # Initialize model and processor
    model, processor = load_model_and_processor()
    if model is None or processor is None:
        st.error("Failed to load model. Please check if model files exist.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Data Analysis"])
    
    if page == "Single Prediction":
        st.header("Single Sample Prediction")
        st.write("Enter spectral values manually or upload a CSV file with a single row.")
        
        upload_method = st.radio("Choose input method:", ["Manual Entry", "File Upload"])
        
        if upload_method == "Manual Entry":
            # Create 448 number inputs in a more compact way
            num_bands = 448
            cols_per_row = 8
            num_rows = num_bands // cols_per_row
            
            spectral_values = []
            for i in range(num_rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    val = cols[j].number_input(f"Band {idx+1}", value=0.0, format="%.4f", key=f"band_{idx}")
                    spectral_values.append(val)
            
            if st.button("Predict"):
                try:
                    X = np.array(spectral_values).reshape(1, -1)
                    X_scaled = processor.scale_features(X)
                    y_scaled = model.model.predict(X_scaled)
                    y_pred = processor.inverse_transform_target(y_scaled)
                    
                    st.success(f"Predicted DON Concentration: {y_pred[0]:.2f} ppb")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        else:  # File Upload
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    if data.shape[0] != 1:
                        st.error("Please upload a CSV file with exactly one row.")
                        return
                    
                    X = data.values.reshape(1, -1)
                    X_scaled = processor.scale_features(X)
                    y_scaled = model.model.predict(X_scaled)
                    y_pred = processor.inverse_transform_target(y_scaled)
                    
                    st.success(f"Predicted DON Concentration: {y_pred[0]:.2f} ppb")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    elif page == "Batch Prediction":
        st.header("Batch Prediction")
        st.write("Upload a CSV file with multiple samples for batch prediction.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.write(data.head())
                
                if st.button("Run Batch Prediction"):
                    X = data.values
                    X_scaled = processor.scale_features(X)
                    y_scaled = model.model.predict(X_scaled)
                    predictions = processor.inverse_transform_target(y_scaled)
                    
                    results_df = pd.DataFrame({
                        'Sample_ID': range(1, len(predictions) + 1),
                        'DON_Concentration_ppb': predictions
                    })
                    
                    st.write("Prediction Results:")
                    st.write(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Plot distribution of predictions
                    fig, ax = plt.subplots()
                    ax.hist(predictions, bins=20)
                    ax.set_xlabel('DON Concentration (ppb)')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Predicted DON Concentrations')
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Data Analysis
        st.header("Data Analysis")
        st.write("Upload your data for quality analysis and visualization.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                analyzer = DataQualityAnalyzer()
                
                # Data quality metrics
                quality_metrics = analyzer.check_data_quality(data)
                st.subheader("Data Quality Metrics")
                for metric, value in quality_metrics.items():
                    st.write(f"{metric}: {value}")
                
                # Spectral curves
                st.subheader("Spectral Curves")
                fig = plt.figure(figsize=(12, 6))
                analyzer.plot_spectral_curves(data)
                st.pyplot(fig)
                
                # Quality summary plots
                st.subheader("Quality Summary Plots")
                fig = plt.figure(figsize=(15, 12))
                analyzer.plot_quality_summary(data)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error analyzing data: {str(e)}")

if __name__ == "__main__":
    main() 