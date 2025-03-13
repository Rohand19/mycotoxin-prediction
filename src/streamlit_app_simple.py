"""Simplified Streamlit app for DON concentration prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from io import StringIO

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import only what we need
from src.preprocessing.data_processor import DataProcessor
from src.utils.data_quality import DataQualityAnalyzer

def main():
    """Main function for the simplified Streamlit app."""
    st.title("DON Concentration Predictor (Simplified Version)")
    st.write("""
    This is a simplified version of the DON Concentration Predictor application.
    The full model is not loaded to avoid TensorFlow initialization issues.
    """)
    
    st.info("⚠️ This version does not make actual predictions but demonstrates the app's interface.")
    
    # Load only the data processor for data analysis
    try:
        processor = DataProcessor()
        processor_loaded = True
        st.success("Data processor loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data processor: {str(e)}")
        processor_loaded = False
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Data Analysis"])
    
    if page == "Single Prediction":
        st.header("Single Sample Prediction (Demo)")
        st.write("Enter spectral values manually or upload a CSV file with a single row.")
        
        upload_method = st.radio("Choose input method:", ["Manual Entry", "File Upload"])
        
        if upload_method == "Manual Entry":
            # Create a simplified input form with fewer inputs
            num_bands = 10  # Just show 10 instead of 448 for demo
            cols_per_row = 5
            num_rows = num_bands // cols_per_row
            
            st.write("Demo: Showing only 10 of 448 spectral bands for simplicity")
            
            spectral_values = []
            for i in range(num_rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    val = cols[j].number_input(f"Band {idx+1}", value=0.5, format="%.4f", key=f"band_{idx}")
                    spectral_values.append(val)
            
            if st.button("Predict (Demo)"):
                # Simulate prediction
                with st.spinner("Simulating prediction..."):
                    import time
                    time.sleep(1)  # Simulate processing time
                    
                    # Generate a random prediction
                    pred_value = np.random.uniform(500, 2000)
                    st.success(f"Demo Prediction: {pred_value:.2f} ppb")
                    st.info("Note: This is a simulated prediction, not an actual model result.")
        
        else:  # File Upload
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.write("Data Preview:")
                    st.write(data.head())
                    
                    if st.button("Predict (Demo)"):
                        # Simulate prediction
                        with st.spinner("Simulating prediction..."):
                            import time
                            time.sleep(1)  # Simulate processing time
                            
                            # Generate a random prediction
                            pred_value = np.random.uniform(500, 2000)
                            st.success(f"Demo Prediction: {pred_value:.2f} ppb")
                            st.info("Note: This is a simulated prediction, not an actual model result.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    elif page == "Batch Prediction":
        st.header("Batch Prediction (Demo)")
        st.write("Upload a CSV file with multiple samples for batch prediction.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.write(data.head())
                
                if st.button("Run Batch Prediction (Demo)"):
                    # Simulate batch prediction
                    with st.spinner("Simulating batch prediction..."):
                        import time
                        time.sleep(2)  # Simulate processing time
                        
                        # Generate random predictions
                        num_samples = min(len(data), 10)  # Limit to 10 samples for demo
                        predictions = np.random.uniform(500, 2000, size=num_samples)
                        
                        results_df = pd.DataFrame({
                            'Sample_ID': range(1, num_samples + 1),
                            'DON_Concentration_ppb': predictions
                        })
                        
                        st.write("Demo Prediction Results:")
                        st.write(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "demo_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                        # Plot distribution of predictions
                        fig, ax = plt.subplots()
                        ax.hist(predictions, bins=10)
                        ax.set_xlabel('DON Concentration (ppb)')
                        ax.set_ylabel('Count')
                        ax.set_title('Distribution of Demo Predicted DON Concentrations')
                        st.pyplot(fig)
                        
                        st.info("Note: These are simulated predictions, not actual model results.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Data Analysis
        st.header("Data Analysis")
        st.write("Upload your data for quality analysis and visualization.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.write(data.head())
                
                # This part can actually work without the model
                if processor_loaded:
                    analyzer = DataQualityAnalyzer()
                    
                    # Basic data statistics
                    st.subheader("Basic Statistics")
                    st.write(data.describe())
                    
                    # Data visualization
                    st.subheader("Data Visualization")
                    
                    # Plot a few columns as an example
                    if data.shape[1] > 5:
                        sample_cols = data.columns[:5]  # First 5 columns
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for col in sample_cols:
                            if pd.api.types.is_numeric_dtype(data[col]):
                                ax.plot(data[col].values, label=col)
                        ax.legend()
                        ax.set_title('Sample Spectral Curves (First 5 Bands)')
                        ax.set_xlabel('Sample Index')
                        ax.set_ylabel('Value')
                        st.pyplot(fig)
                    
                    # Correlation heatmap
                    st.subheader("Correlation Heatmap")
                    if data.shape[1] <= 20:  # Only show if not too many columns
                        corr = data.corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        cax = ax.matshow(corr, cmap='coolwarm')
                        fig.colorbar(cax)
                        st.pyplot(fig)
                    else:
                        st.info("Correlation heatmap not shown due to large number of columns.")
                    
                    st.info("Note: This is a simplified analysis. The full app provides more detailed analysis.")
                else:
                    st.error("Data processor not loaded. Cannot perform analysis.")
            except Exception as e:
                st.error(f"Error analyzing data: {str(e)}")

if __name__ == "__main__":
    main() 