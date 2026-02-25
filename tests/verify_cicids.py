import os
import pickle
import sys

import torch


# Mock Streamlit to capture errors
class MockStreamlit:
    def __init__(self, log_file):
        self.log_file = log_file

    def cache_resource(self, func):
        return func

    def cache_data(self, func):
        return func

    def error(self, msg):
        with open(self.log_file, "a") as f:
            f.write(f"ST_ERROR: {msg}\n")

    def warning(self, msg):
        with open(self.log_file, "a") as f:
            f.write(f"ST_WARNING: {msg}\n")

    def info(self, msg):
        pass


# Initialize mock
log_file = "cicids_verification_log.txt"
sys.modules["streamlit"] = MockStreamlit(log_file)

# Add project root to path
sys.path.append(os.getcwd())

from frontend.utils import MODELS_DIR, load_feature_columns, load_model_and_scaler


def verify_cicids():
    with open(log_file, "w") as f:
        f.write("Verifying CICIDS2017 Models...\n")
        dataset = "CICIDS2017"

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            f.write(f"Device: {device}\n")
        except Exception as e:
            f.write(f"Device Error: {e}\n")
            return

        # Check Feature Columns
        try:
            cols_path = os.path.join(MODELS_DIR, "cicids2017_feature_cols.pkl")
            if os.path.exists(cols_path):
                with open(cols_path, "rb") as cf:
                    cols = pickle.load(cf)
                    f.write(f"Feature Cols Count: {len(cols)}\n")
            else:
                f.write("Feature Cols file not found.\n")
        except Exception as e:
            f.write(f"Error loading feature cols: {e}\n")

        # 1. Transformer
        f.write("\n--- Loading Transformer ---\n")
        try:
            model, scaler, encoders = load_model_and_scaler(
                "Transformer", dataset, device
            )
            if model:
                f.write("Success: Transformer loaded.\n")
            else:
                f.write("Failed: Transformer returned None.\n")
        except Exception as e:
            f.write(f"Exception loading Transformer: {e}\n")

        # 2. Autoencoder
        f.write("\n--- Loading Autoencoder ---\n")
        try:
            model, scaler, encoders = load_model_and_scaler(
                "Autoencoder", dataset, device
            )
            if model:
                f.write("Success: Autoencoder loaded.\n")
            else:
                f.write("Failed: Autoencoder returned None.\n")
        except Exception as e:
            f.write(f"Exception loading Autoencoder: {e}\n")

        # 3. CNN
        f.write("\n--- Loading CNN ---\n")
        try:
            model, scaler, encoders = load_model_and_scaler("CNN", dataset, device)
            if model:
                f.write("Success: CNN loaded.\n")
            else:
                f.write("Failed: CNN returned None.\n")
        except Exception as e:
            f.write(f"Exception loading CNN: {e}\n")

    print(f"Results written to {log_file}")


if __name__ == "__main__":
    verify_cicids()
