from data_preprocessing import load_data, preprocess_data
from model import build_model, train_model
from eval import evaluate_model, plot_results
from sklearn.model_selection import train_test_split
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import gc
import matplotlib.pyplot as plt

# Load and preprocess data
print("Starting data loading...")
start_time = time.time()
df = load_data('data/corn_hyperspectral.csv')
print(f"Data loaded in {time.time() - start_time:.2f} seconds. Shape: {df.shape}")

print("Starting preprocessing...")
start_time = time.time()
X_scaled, y_scaled, X_scaler, y_scaler = preprocess_data(df)
print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds. X shape: {X_scaled.shape}")

# Print data statistics
print("\nData statistics after scaling:")
print(f"X - Mean: {np.mean(X_scaled):.4f}, Std: {np.std(X_scaled):.4f}")
print(f"y - Mean: {np.mean(y_scaled):.4f}, Std: {np.std(y_scaled):.4f}")

# Split data with stratification
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, 
    test_size=0.2, 
    random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Build and train model
print("\nBuilding model...")
model = build_model(input_shape=X_train.shape[1])
print("Model built successfully. Starting training...")

history = train_model(model, X_train, y_train, 
                     epochs=100,  # Max epochs (with early stopping)
                     batch_size=32)

print("\nTraining completed. Starting evaluation...")

# Clear memory before evaluation
gc.collect()
tf.keras.backend.clear_session()

# Evaluate model using TensorFlow Dataset
try:
    print("Making predictions on test set...")
    
    # Make predictions in smaller batches
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_dataset = test_dataset.batch(32)
    
    # Make predictions
    y_pred_scaled = []
    total_batches = len(list(test_dataset))
    
    print(f"Processing {total_batches} batches...")
    for i, batch in enumerate(test_dataset):
        # Make prediction
        pred = model(batch, training=False)
        y_pred_scaled.append(pred.numpy())
        print(f"Processed batch {i+1}/{total_batches}")
        
        # Clear memory after each batch
        gc.collect()
    
    # Combine predictions
    print("Combining predictions...")
    y_pred_scaled = np.vstack(y_pred_scaled)
    
    print("Converting predictions back to original scale...")
    # Convert predictions back to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics on original scale
    print("Calculating metrics...")
    mse = np.mean((y_test_original - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_original - y_pred))
    r2 = 1 - (np.sum((y_test_original - y_pred) ** 2) / np.sum((y_test_original - np.mean(y_test_original)) ** 2))
    
    print("\nTest Set Metrics (on original scale):")
    print(f"RMSE: {rmse:.2f} ppb")
    print(f"MAE: {mae:.2f} ppb")
    print(f"R²: {r2:.4f}")
    
    # Save predictions and training history
    print("\nSaving results...")
    results_df = pd.DataFrame({
        'Actual': y_test_original.ravel(),
        'Predicted': y_pred.ravel(),
        'Error': y_test_original.ravel() - y_pred.ravel()
    })
    results_df.to_csv('prediction_results.csv', index=False)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training History
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot 2: Predictions vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_original, y_pred, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Values (ppb)')
    plt.ylabel('Predicted Values (ppb)')
    plt.title('Predictions vs Actual Values')
    
    plt.tight_layout()
    plt.savefig('model_analysis.png')
    plt.close()
    
    # Save the model and scalers
    print("Saving model and scalers...")
    model.save('don_prediction_model.keras')
    import joblib
    joblib.dump(X_scaler, 'X_scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')
    print("Process completed successfully!")

except Exception as e:
    print(f"Error during evaluation: {str(e)}")
    import traceback
    print("Full traceback:")
    print(traceback.format_exc())