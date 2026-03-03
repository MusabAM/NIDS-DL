import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "raw", "cicids2018")
output_scaler_path = os.path.join(
    base_dir, "results", "models", "cicids2018_scaler.pkl"
)
output_features_path = os.path.join(
    base_dir, "results", "models", "cicids2018_feature_cols.pkl"
)

# Columns to drop (metadata and labels)
DROP_COLS = {
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
    "binary_label",
}


def generate_scaler():
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}!")
        return

    print(
        f"Found {len(csv_files)} CSV files in {data_dir}. Processing with partial_fit..."
    )

    scaler = StandardScaler()
    feature_cols = None

    for file in csv_files:
        print(f"Processing {os.path.basename(file)}...")
        try:
            # Read in chunks to avoid memory errors with massive CSV files
            chunk_iterator = pd.read_csv(
                file, chunksize=250000, low_memory=False, on_bad_lines="skip"
            )
            for chunk_idx, chunk in enumerate(chunk_iterator):
                # Clean up column names strictly
                chunk.columns = chunk.columns.str.strip()

                # Drop metadata cols
                cols_to_drop = [c for c in DROP_COLS if c in chunk.columns]
                df_features = chunk.drop(columns=cols_to_drop)

                # Convert to numeric
                for col in df_features.columns:
                    df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

                # Clean broken numbers
                df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_features.fillna(0, inplace=True)

                if feature_cols is None:
                    # Lock in the feature columns based on the very first valid chunk
                    feature_cols = df_features.columns.tolist()

                # Ensure every subsequent chunk has the exact same columns
                df_features = df_features.reindex(columns=feature_cols, fill_value=0)

                # Incrementally fit the StandardScaler
                scaler.partial_fit(df_features)
                print(f"  - Chunk {chunk_idx + 1} processed.")
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {e}")

    if feature_cols is None:
        print("No features found to process!")
        return

    print(f"Number of features extracted: {len(feature_cols)}")

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
