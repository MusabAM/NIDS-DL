import sys
import os
import torch
import pickle

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from backend.utils import load_model_and_scaler, DATASET_CONFIGS, MODELS_DIR

datasets = ["NSL-KDD", "CICIDS2018", "CICIDS2017", "UNSW-NB15"]
model_types = ["CNN", "LSTM", "Transformer", "Autoencoder"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting Detailed Verification (Device: {device})...")
print(f"MODELS_DIR: {MODELS_DIR}")
print("-" * 50)

for ds in datasets:
    config = DATASET_CONFIGS.get(ds)
    if config is None:
        print(f"FAILED: Dataset {ds} not found in DATASET_CONFIGS")
        continue
        
    for mt in model_types:
        rel_path = config["model_files"].get(mt)
        if not rel_path:
            print(f"SKIP: {ds} - {mt} (No model file configured)")
            continue
            
        full_path = os.path.join(MODELS_DIR, rel_path)
        exists = os.path.exists(full_path)
        
        print(f"{ds} - {mt}:")
        print(f"  Path: {rel_path}")
        print(f"  Full: {full_path}")
        print(f"  Exists: {exists}")
        
        if not exists:
            print(f"  ERROR: File not found")
            continue
            
        try:
            model, scaler, encoders = load_model_and_scaler(mt, ds, device)
            if model is not None:
                print("  LOAD: SUCCESS")
            else:
                print("  LOAD: FAILED (Returned None)")
        except Exception as e:
            print(f"  LOAD: ERROR ({str(e)})")

print("-" * 50)
