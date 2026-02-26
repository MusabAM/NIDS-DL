import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Set paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "raw", "cicids2018", "processed_data.csv")
output_scaler_path = os.path.join(base_dir, "results", "models", "cicids2018_scaler.pkl")
output_features_path = os.path.join(base_dir, "results", "models", "cicids2018_feature_cols.pkl")

# Columns to drop (metadata and labels)
DROP_COLS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
    "binary_label",
]

def generate_scaler():
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print("Data file not found!")
        return

    # Load a portion if large, or full if possible
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # Drop metadata
    df_features = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    
    # Convert to numeric and handle Inf/NaN
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
    
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.fillna(0, inplace=True)
    
    feature_cols = df_features.columns.tolist()
    print(f"Number of features: {len(feature_cols)}")

    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(df_features)

    # Save Scaler
    os.makedirs(os.path.dirname(output_scaler_path), exist_ok=True)
    with open(output_scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {output_scaler_path}")

    # Save Feature Columns
    with open(output_features_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"Saved feature columns to {output_features_path}")

if __name__ == "__main__":
    generate_scaler()
