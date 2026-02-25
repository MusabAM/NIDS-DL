import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Config
DATA_DIR = Path("data/raw")
MODELS_DIR = Path("results/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# UNSW-NB15 Config
UNSW_TRAIN = DATA_DIR / "unsw-nb15/UNSW_NB15_training-set.csv"
UNSW_CATEGORICAL_COLS = ["proto", "service", "state"]
UNSW_DROP_COLS = ["id", "attack_cat", "label"]

# CICIDS2017 Config
CICIDS_DIR = DATA_DIR / "cicids2017"


def generate_unsw_resources():
    print("Generating UNSW-NB15 resources...")
    if not UNSW_TRAIN.exists():
        print(f"Error: {UNSW_TRAIN} not found.")
        return

    # Load data
    try:
        df = pd.read_csv(UNSW_TRAIN)
    except Exception as e:
        print(f"Error reading UNSW data: {e}")
        return

    # Encoders
    label_encoders = {}

    # We need to handle categorical columns
    # In the training script, they filled NaN with 'unknown' and converted to string

    for col in UNSW_CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            # Handle potential mixed types
            df[col] = df[col].fillna("unknown").astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Save Encoders
    with open(MODELS_DIR / "unsw_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    print(f"Saved unsw_encoders.pkl to {MODELS_DIR}")

    # Drop target and ID
    # Also drop categorical columns if they are not used as features?
    # Usually we use the encoded versions. So we keep them.
    # The training script drops them from "X" BEFORE encoding if they are in DROP_COLS.
    # 'proto', 'service', 'state' are features.

    y = df["label"].values
    X = df.drop(columns=[c for c in UNSW_DROP_COLS if c in df.columns], errors="ignore")

    # Save Feature Columns (the ones remaining after drop)
    feature_cols = list(X.columns)
    with open(MODELS_DIR / "unsw_feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print(
        f"Saved unsw_feature_cols.pkl to {MODELS_DIR} with {len(feature_cols)} features."
    )

    # Scaler
    # We need to handle encoded categorical columns too
    # The training script scales EVERYTHING in X.

    scaler = StandardScaler()
    try:
        scaler.fit(X)
    except Exception as e:
        print(f"Error fitting scaler for UNSW: {e}")
        return

    with open(MODELS_DIR / "unsw_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved unsw_scaler.pkl to {MODELS_DIR}")


def generate_cicids2017_resources():
    print("\nGenerating CICIDS2017 resources...")

    # We need to find the files
    # The notebook used: all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

    pattern = str(CICIDS_DIR / "*.csv")
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"Error: No CICIDS2017 files found in {CICIDS_DIR}")
        return

    print(f"Found {len(all_files)} files. Using a sample for scaler fitting.")

    li = []

    # Columns to drop based on notebook
    drop_cols = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        "Timestamp",
        "Label",
    ]

    for filename in all_files:
        try:
            print(f"  Reading {os.path.basename(filename)}...")
            # Read first chunk to get sample
            # 50k rows per file should be enough to estimate mean/var
            df_chunk = pd.read_csv(filename, index_col=None, header=0, nrows=50000)
            li.append(df_chunk)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

    if not li:
        print("No data loaded for CICIDS2017.")
        return

    df = pd.concat(li, axis=0, ignore_index=True)

    # Cleaning steps from notebook
    # 1. Strip whitespace
    df.columns = df.columns.str.strip()

    # 2. Replace Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 3. Drop cols
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Save Feature Columns
    feature_cols = list(X.columns)
    with open(MODELS_DIR / "cicids2017_feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"Saved cicids2017_feature_cols.pkl with {len(feature_cols)} features.")

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X)

    with open(MODELS_DIR / "cicids2017_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved cicids2017_scaler.pkl")


if __name__ == "__main__":
    # Ensure raw data dir exists
    if not DATA_DIR.exists():
        print(f"Warning: Data directory {DATA_DIR.absolute()} does not exist.")

    generate_unsw_resources()
    generate_cicids2017_resources()
    print("Done.")
