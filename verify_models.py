import sys
import os
import torch

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from backend.utils import load_model_and_scaler, DATASET_CONFIGS

datasets = ["NSL-KDD", "CICIDS2018", "CICIDS2017", "UNSW-NB15"]
model_types = ["CNN", "LSTM", "Transformer", "Autoencoder"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

success_count = 0
fail_count = 0

print(f"Starting Path Verification (Device: {device})...")
print("-" * 30)

for ds in datasets:
    for mt in model_types:
        try:
            print(f"Testing {ds} - {mt}...", end=" ")
            # Correct order: model_name, dataset, device
            model, scaler, encoders = load_model_and_scaler(mt, ds, device)
            if model is not None:
                print("SUCCESS")
                success_count += 1
            else:
                # Some datasets might not have all model types yet? 
                # Actually, according to CONFIGs, most should have them.
                print("FAILED (Model is None - check if path exists)")
                fail_count += 1
        except Exception as e:
            print(f"FAILED (Error: {str(e)})")
            fail_count += 1

print("-" * 30)
print(f"Verification Finished: {success_count} Successes, {fail_count} Failures")

if fail_count > 0:
    # Check if failures are expected (e.g. missing files in workspace)
    pass
