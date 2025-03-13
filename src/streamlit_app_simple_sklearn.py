"""Streamlit app for DON concentration prediction using scikit-learn.

This version completely avoids TensorFlow to ensure reliability on all platforms.
"""

import os
import sys
import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our simple model and other modules
from src.models.simple_predictor import SimplePredictor
from src.preprocessing.data_processor import DataProcessor
from src.utils.data_quality import DataQualityAnalyzer


def load_model_and_processor():
    """Load the model and data processor with improved error handling."""
    try:
        # We'll still try to load the processor for data analysis
        try:
            st.info("Loading data processor...")
            processor = DataProcessor()
            processor.load_scalers("models/X_scaler_real.pkl", "models/y_scaler_real.pkl")
            st.success("Data processor loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load data processor: {str(e)}")
            st.info("Using internal scaling instead.")
            processor = None

        st.info("Creating scikit-learn model trained on real data...")
        start_time = time.time()

        # Create a simple model trained on real data
        model = SimplePredictor.load_or_create("models/rf_model_real_data.joblib", train_on_real_data=True)

        end_time = time.time()

        if model is None:
            st.error("Failed to create model.")
            return None, processor

        st.success(f"Model created successfully in {end_time - start_time:.2f} seconds!")
        return model, processor
    except Exception as e:
        st.error(f"Error during loading: {str(e)}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None


def process_uploaded_file(uploaded_file, predictor):
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)

        # Check for non-numeric columns
        non_numeric_cols = data.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric_cols:
            st.warning(f"Warning: Non-numeric columns detected: {', '.join(non_numeric_cols)}. These will be ignored.")
            # Filter to keep only numeric columns
            data = data.select_dtypes(include=["number"])

        if data.empty:
            st.error("Error: No numeric data found in the uploaded file.")
            return None

        # Check feature count
        expected_features = 448
        feature_count = data.shape[1]
        if feature_count != expected_features:
            if feature_count > expected_features:
                st.info(
                    f"Note: Your data has {feature_count} features, but the model expects {expected_features}. The extra features will be ignored."
                )
            else:
                st.info(
                    f"Note: Your data has {feature_count} features, but the model expects {expected_features}. Missing features will be filled with zeros."
                )

        # Make predictions
        try:
            predictions = predictor.predict(data.values)
            predictions = predictor.inverse_transform_target(predictions)

            # Create a DataFrame with the predictions
            result_df = pd.DataFrame(
                {
                    "Sample": range(1, len(predictions) + 1),
                    "DON Concentration (ppb)": predictions.flatten(),
                }
            )

            return result_df
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info(
                "Please ensure your data has the correct format: numeric values only, with each row representing a sample."
            )
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file is a valid CSV with numeric data only.")
        return None


def main():
    """Main function for the Streamlit app."""
    st.title("DON Concentration Predictor (scikit-learn Version)")
    st.write(
        """
    This application predicts DON (Deoxynivalenol) concentration in corn samples
    using hyperspectral imaging data. This version uses scikit-learn instead of
    TensorFlow to ensure reliability on all platforms.
    """
    )

    st.info(
        """
    This model is trained on real hyperspectral data from corn samples,
    providing more accurate predictions of DON concentration.
    """
    )

    # Initialize model and processor
    model, processor = load_model_and_processor()

    if model is None:
        st.error("Failed to load model. Please check the error messages above.")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Data Analysis"])

    if page == "Single Prediction":
        st.header("Single Sample Prediction")
        st.write("Enter spectral values manually or upload a CSV file with a single row.")

        upload_method = st.radio("Choose input method:", ["Manual Entry", "File Upload"])

        if upload_method == "Manual Entry":
            # Create a simplified input form with fewer inputs
            num_bands = 10  # Just show 10 instead of 448 for demo
            cols_per_row = 5
            num_rows = num_bands // cols_per_row

            st.write("Demo: Showing only 10 of 448 spectral bands for simplicity")
            st.write("(The remaining bands will be set to default values)")

            spectral_values = [0.5] * 448  # Default values
            for i in range(num_rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    val = cols[j].number_input(
                        f"Band {idx+1}",
                        value=0.5,
                        format="%.4f",
                        key=f"band_{idx}",
                    )
                    spectral_values[idx] = val

            if st.button("Predict"):
                try:
                    with st.spinner("Making prediction..."):
                        X = np.array(spectral_values).reshape(1, -1)

                        # Use the model's internal scaling instead of the processor
                        y_pred = model.predict(X)
                        y_pred = model.inverse_transform_target(y_pred)

                        st.success(f"Predicted DON Concentration: {y_pred[0][0]:.2f} ppb")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

        else:  # File Upload
            uploaded_file = st.file_uploader("Upload a CSV file with a single sample", type="csv")
            if uploaded_file is not None:
                try:
                    # Read the CSV file
                    data = pd.read_csv(uploaded_file)

                    # Check for non-numeric columns
                    non_numeric_cols = data.select_dtypes(exclude=["number"]).columns.tolist()
                    if non_numeric_cols:
                        st.warning(
                            f"Warning: Non-numeric columns detected: {', '.join(non_numeric_cols)}. These will be ignored."
                        )
                        # Filter to keep only numeric columns
                        data = data.select_dtypes(include=["number"])

                    if data.empty:
                        st.error("Error: No numeric data found in the uploaded file.")
                    elif len(data) > 1:
                        st.warning(
                            "Warning: The uploaded file contains multiple samples. Only the first sample will be used."
                        )
                        sample_values = data.iloc[0].values
                    else:
                        sample_values = data.iloc[0].values

                    # Display the spectral values
                    if "sample_values" in locals():
                        st.write(f"Loaded {len(sample_values)} spectral values from the file.")

                        # Inform user if feature count doesn't match model expectations
                        expected_features = 448
                        if len(sample_values) != expected_features:
                            if len(sample_values) > expected_features:
                                st.info(
                                    f"Note: Your data has {len(sample_values)} features, but the model expects {expected_features}. The extra features will be ignored."
                                )
                            else:
                                st.info(
                                    f"Note: Your data has {len(sample_values)} features, but the model expects {expected_features}. Missing features will be filled with zeros."
                                )

                        # Make prediction
                        with st.spinner("Making prediction..."):
                            X = np.array(sample_values).reshape(1, -1)

                            # Use the model's internal scaling instead of the processor
                            y_pred = model.predict(X)
                            prediction = model.inverse_transform_target(y_pred)

                            st.success(f"Predicted DON Concentration: {prediction[0][0]:.2f} ppb")

                            # Plot the spectral curve
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(range(len(sample_values)), sample_values)
                            ax.set_xlabel("Spectral Band")
                            ax.set_ylabel("Reflectance")
                            ax.set_title("Spectral Curve of Sample")
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please ensure your file is a valid CSV with numeric data only.")

    elif page == "Batch Prediction":
        st.header("Batch Prediction")
        st.write("Upload a CSV file with spectral data for multiple samples.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_file")

        if uploaded_file is not None:
            st.info("Processing file... This may take a moment.")

            # Process the uploaded file
            result_df = process_uploaded_file(uploaded_file, model)

            if result_df is not None:
                # Display the predictions
                st.subheader("Predictions")
                st.dataframe(result_df)

                # Allow downloading the predictions
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="don_predictions.csv",
                    mime="text/csv",
                )

    else:  # Data Analysis
        st.header("Data Analysis")
        st.write(
            """
        Upload your hyperspectral data for quality analysis and visualization.
        This tool will help you understand the characteristics of your data and identify potential issues.
        """
        )

        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                # Read the CSV file
                data = pd.read_csv(uploaded_file)

                # Display basic information about the data
                st.subheader("Data Overview")
                st.write(f"Total rows: {data.shape[0]}, Total columns: {data.shape[1]}")

                # Check for non-numeric columns
                non_numeric_cols = data.select_dtypes(exclude=["number"]).columns.tolist()
                if non_numeric_cols:
                    st.info(
                        f"Non-numeric columns detected: {', '.join(non_numeric_cols)}. These will be excluded from numerical analysis."
                    )

                # Display a sample of the data
                st.write("Data Preview:")
                st.dataframe(data.head())

                # Create a DataQualityAnalyzer
                analyzer = DataQualityAnalyzer()

                # Basic data statistics
                st.subheader("Basic Statistics")
                numeric_data = data.select_dtypes(include=["number"])
                if not numeric_data.empty:
                    # Show summary statistics
                    st.write(numeric_data.describe())

                    # Show distribution of values
                    st.subheader("Distribution of Values")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Flatten all numeric values and plot histogram
                    all_values = numeric_data.values.flatten()
                    all_values = all_values[~np.isnan(all_values)]  # Remove NaNs

                    sns.histplot(all_values, kde=True, ax=ax)
                    ax.set_title("Distribution of All Numeric Values")
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                else:
                    st.warning("No numeric data found for statistical analysis.")

                # Try to use the analyzer's methods for more detailed analysis
                try:
                    # Data quality metrics
                    quality_metrics = analyzer.check_data_quality(data)

                    st.subheader("Data Quality Metrics")
                    # Create a more user-friendly display of metrics
                    metrics_df = pd.DataFrame(
                        {
                            "Metric": list(quality_metrics.keys()),
                            "Value": list(quality_metrics.values()),
                        }
                    )
                    st.table(metrics_df)

                    # Highlight key insights
                    if "missing_values" in quality_metrics and quality_metrics["missing_values"] > 0:
                        st.warning(
                            f"⚠️ Your data contains {quality_metrics['missing_values']} missing values ({quality_metrics['missing_percentage']:.2f}%)."
                        )

                    if "negative_values" in quality_metrics and quality_metrics["negative_values"] > 0:
                        st.warning(
                            f"⚠️ Your data contains {quality_metrics['negative_values']} negative values, which may be unusual for reflectance data."
                        )

                    if "potential_outliers" in quality_metrics and quality_metrics["potential_outliers"] > 0:
                        st.warning(
                            f"⚠️ Detected {quality_metrics['potential_outliers']} potential outliers in your data."
                        )

                    # Spectral curves visualization
                    st.subheader("Spectral Curves")
                    st.write(
                        "This plot shows the average spectral curve across all samples, with the shaded area representing one standard deviation."
                    )

                    # Create a figure for the spectral curves
                    fig = plt.figure(figsize=(12, 6))
                    analyzer.plot_spectral_curves(data)
                    st.pyplot(fig)

                    # Quality summary plots
                    st.subheader("Quality Summary Plots")
                    st.write("These plots provide insights into the distribution and relationships in your data.")

                    # Create a figure for the quality summary
                    fig = plt.figure(figsize=(15, 12))
                    analyzer.plot_quality_summary(data)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Could not perform full analysis: {str(e)}")
                    st.write("Detailed error:", str(e))
                    import traceback

                    st.code(traceback.format_exc())

                    # Fallback: simple visualization if the full analysis fails
                    st.subheader("Simple Data Visualization")

                    # Plot a few columns as an example
                    numeric_data = data.select_dtypes(include=["number"])
                    if not numeric_data.empty:
                        if numeric_data.shape[1] > 5:
                            # Take evenly spaced columns for better representation
                            indices = np.linspace(0, numeric_data.shape[1] - 1, 5, dtype=int)
                            sample_cols = numeric_data.columns[indices]

                            fig, ax = plt.subplots(figsize=(10, 6))
                            for col in sample_cols:
                                ax.plot(
                                    numeric_data[col].values[:20],
                                    label=str(col),
                                )  # Plot first 20 values
                            ax.legend()
                            ax.set_title("Sample Spectral Values (First 20 Samples)")
                            ax.set_xlabel("Sample Index")
                            ax.set_ylabel("Value")
                            st.pyplot(fig)

                        # Simple correlation heatmap
                        st.subheader("Simple Correlation Heatmap")
                        if numeric_data.shape[1] <= 20:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            corr = numeric_data.corr()
                            sns.heatmap(
                                corr,
                                cmap="coolwarm",
                                ax=ax,
                                vmin=-1,
                                vmax=1,
                                center=0,
                            )
                            ax.set_title("Correlation Between Features")
                            st.pyplot(fig)
                        else:
                            st.info("Correlation heatmap not shown due to large number of columns.")
                    else:
                        st.warning("No numeric data available for visualization.")

            except Exception as e:
                st.error(f"Error analyzing data: {str(e)}")
                st.write("Please ensure your file is a valid CSV with properly formatted data.")


if __name__ == "__main__":
    main()
