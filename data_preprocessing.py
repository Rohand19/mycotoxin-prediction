import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Split features and target
    X = df.drop(['hsi_id', 'vomitoxin_ppb'], axis=1)
    y = df['vomitoxin_ppb']
    
    # Scale features
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    
    # Scale target (reshape needed for 1D array)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    return X_scaled, y_scaled, X_scaler, y_scaler