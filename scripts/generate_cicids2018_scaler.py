"""
Generate and save a StandardScaler for CICIDS2018 dataset.

Uses chunked CSV reading to avoid MemoryError on large datasets.

Usage:
    python scripts/generate_cicids2018_scaler.py
"""

import os
import sys
import glob
import pickle
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cicids2018")
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "models", "cicids2018_scaler.pkl"
)

# Metadata columns to drop (same as training notebooks)
DROP_COLS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
]


def create_binary_label(label):
    """Convert label to binary: 0=Benign, 1=Attack."""
    if isinstance(label, str) and "BENIGN" in label.upper():
        return 0
    return 1


def load_file_chunked(filename, samples_per_file=50000, chunk_size=50000):
    """Load a CSV file in chunks and sample rows to save memory."""
    sampled_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(filename, chunksize=chunk_size, low_memory=True):
        chunk.columns = chunk.columns.str.strip()

        # Create binary label
        if "Label" in chunk.columns:
            chunk["binary_label"] = chunk["Label"].apply(create_binary_label)
            chunk.drop(columns=["Label"], inplace=True, errors="ignore")

        # Drop metadata columns
        for col in DROP_COLS:
            if col in chunk.columns:
                chunk.drop(columns=[col], inplace=True, errors="ignore")

        # Convert to numeric
        for col in chunk.columns:
            if col != "binary_label":
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Clean NaN/Inf
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        total_rows += len(chunk)

        # Sample from this chunk proportionally
        n_sample = min(len(chunk), max(1000, samples_per_file // 10))
        if len(chunk) > 0:
            sampled_chunks.append(chunk.sample(n=min(n_sample, len(chunk)), random_state=42))

        del chunk
        gc.collect()

        # Stop after getting enough samples
        if sum(len(c) for c in sampled_chunks) >= samples_per_file:
            break

    if not sampled_chunks:
        return None, total_rows

    result = pd.concat(sampled_chunks, ignore_index=True)

    # Final sample to target
    if len(result) > samples_per_file:
        result = result.sample(n=samples_per_file, random_state=42)

    del sampled_chunks
    gc.collect()
    return result, total_rows


def main():
    print("=" * 60)
    print("CICIDS2018 Scaler Generator (Memory-Efficient)")
    print("=" * 60)

    # Find CSV files
    all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    all_files = sorted(all_files, key=lambda x: os.path.getsize(x))

    if not all_files:
        print(f"ERROR: No CSV files found in {DATA_PATH}")
        sys.exit(1)

    print(f"Found {len(all_files)} files.\n")

    # Load & sample per file
    SAMPLES_PER_FILE = 50000
    li = []

    for filename in all_files:
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        if file_size_mb > 1000:
            print(f"  Skipping {os.path.basename(filename)} ({file_size_mb:.0f}MB - too large)")
            continue
        print(f"  Processing {os.path.basename(filename)} ({file_size_mb:.0f}MB)...", end=" ", flush=True)

        df_sampled, total = load_file_chunked(filename, SAMPLES_PER_FILE)
        if df_sampled is not None:
            li.append(df_sampled)
            print(f"sampled {len(df_sampled):,} from {total:,} rows")
        else:
            print("no valid data")

        gc.collect()

    df = pd.concat(li, axis=0, ignore_index=True)
    del li
    gc.collect()
    print(f"\nTotal sampled: {len(df):,} rows")

    # Balance classes
    if "binary_label" in df.columns:
        benign = df[df["binary_label"] == 0]
        attack = df[df["binary_label"] == 1]
        n_per_class = min(len(benign), len(attack), 200000)
        if n_per_class > 0:
            benign = benign.sample(n=n_per_class, random_state=42)
            attack = attack.sample(n=n_per_class, random_state=42)
            df = pd.concat([benign, attack], ignore_index=True)
        print(f"Balanced to {len(df):,} samples ({n_per_class:,} per class)")

    # Get feature columns (Natural Order as in CSV/Notebooks)
    # Exclude binary_label and metadata
    feature_cols = [c for c in df.columns if c != "binary_label"]
    print(f"Number of features: {len(feature_cols)}")

    # Separate features
    X = df[feature_cols].values.astype(np.float32)
    del df
    gc.collect()

    # Fit scaler on all data (this is already sampled)
    scaler = StandardScaler()
    scaler.fit(X)
    print(f"Scaler fitted on {len(X):,} samples")

    # Save scaler
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to: {OUTPUT_PATH}")

    # Also save feature column order
    feature_cols_path = OUTPUT_PATH.replace("_scaler.pkl", "_feature_cols.pkl")
    with open(feature_cols_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"Feature columns saved to: {feature_cols_path}")

    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
