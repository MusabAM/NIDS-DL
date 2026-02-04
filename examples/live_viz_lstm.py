"""
Example: Run live Hilbert map with LSTM model on NSL-KDD test data
"""

from pathlib import Path

from src.visualization.live_hilbert_map import run_with_model

# Paths
MODEL_PATH = "results/models/lstm_nsl_kdd_best.h5"
DATA_PATH = "data/processed/NSL_KDD/Test/y_test.csv"

# Check if files exist
if not Path(MODEL_PATH).exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    print(
        "Please train the LSTM model first using notebooks/02_classical_dl/05_lstm_nsl_kdd.ipynb"
    )
    exit(1)

if not Path(DATA_PATH).exists():
    print(f"❌ Data not found: {DATA_PATH}")
    print("Please ensure test data is preprocessed")
    exit(1)

print("🚀 Starting live visualization with LSTM model...")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Data: {DATA_PATH}")
print("=" * 60)
print("\nControls:")
print("  R - Reset visualization")
print("  ESC - Quit")
print("\nStarting in 3 seconds...")

import time

time.sleep(3)

# Run visualization
run_with_model(
    model_path=MODEL_PATH,
    data_path=DATA_PATH,
    order=7,  # 128x128 grid (16,384 points)
    cell_size=4,  # 4 pixels per cell
)
