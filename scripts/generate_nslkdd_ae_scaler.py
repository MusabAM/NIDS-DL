"""
Generate a scaler for the NSL-KDD Autoencoder.

The NSL-KDD Autoencoder was trained on the 41 raw numeric features
(with categorical columns label-encoded, NOT one-hot encoded),
so it needs its own scaler separate from the CNN/LSTM/Transformer scaler
which operates on 122 one-hot-encoded features.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

TRAIN_DATA_PATH = os.path.join(base_dir, "data", "raw", "nsl-kdd", "train.txt")
OUTPUT_SCALER_PATH = os.path.join(
    base_dir, "results", "models", "autoencoder_nsl_kdd_scaler.pkl"
)
OUTPUT_ENCODERS_PATH = os.path.join(
    base_dir, "results", "models", "autoencoder_nsl_kdd_encoders.pkl"
)

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def generate_scaler():
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Training data not found at {TRAIN_DATA_PATH}")
        return

    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    df = pd.read_csv(TRAIN_DATA_PATH, header=None, names=NSL_KDD_COLUMNS)

    # Drop label and difficulty
    df = df.drop(columns=["label", "difficulty_level"])

    # Label-encode categorical columns (same as autoencoder training)
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Label-encoded '{col}': {len(le.classes_)} classes")

    print(f"Feature count: {df.shape[1]} (should be 41)")

    # Fit scaler on all 41 features
    scaler = StandardScaler()
    scaler.fit(df.values.astype(np.float32))

    print(f"Scaler n_features_in_: {scaler.n_features_in_}")

    # Save scaler
    os.makedirs(os.path.dirname(OUTPUT_SCALER_PATH), exist_ok=True)
    with open(OUTPUT_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {OUTPUT_SCALER_PATH}")

    # Save label encoders
    with open(OUTPUT_ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Saved encoders to {OUTPUT_ENCODERS_PATH}")


if __name__ == "__main__":
    generate_scaler()
