import os
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
log_file = "unsw_verification_log.txt"
sys.modules["streamlit"] = MockStreamlit(log_file)

# Add project root to path
sys.path.append(os.getcwd())

from frontend.utils import load_feature_columns, load_model_and_scaler


def verify_unsw():
    log_file = "unsw_verification_log.txt"
    with open(log_file, "w") as f:
        f.write("Verifying UNSW-NB15 Models...\n")
        dataset = "UNSW-NB15"
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            f.write(f"Device: {device}\n")
        except Exception as e:
            f.write(f"Device Error: {e}\n")
            return

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
    verify_unsw()
