import os
import sys
from unittest.mock import MagicMock

import pandas as pd
import torch


# Mock Streamlit
class MockStreamlit:
    def cache_resource(self, func):
        return func

    def cache_data(self, func):
        return func

    def error(self, msg):
        print(f"ST_ERROR: {msg}")

    def warning(self, msg):
        print(f"ST_WARNING: {msg}")

    def info(self, msg):
        print(f"ST_INFO: {msg}")


sys.modules["streamlit"] = MockStreamlit()

# Now import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from frontend import utils
except ImportError as e:
    print(f"Failed to import frontend.utils: {e}")
    sys.exit(1)


def verify_dataset(name):
    print(f"\nVerifying {name}...")
    print(f"  CWD: {os.getcwd()}")
    print(f"  MODELS_DIR: {utils.MODELS_DIR}")

    config = utils.DATASET_CONFIG.get(name)
    if not config:
        print(f"  Error: No config for {name}")
        return

    # Check Feature Columns
    try:
        cols = utils.load_feature_columns(name)
        print(f"  Feature Cols: Loaded {len(cols)} columns.")
    except Exception as e:
        print(f"  Error loading feature cols: {e}")

    # Check Models
    device = torch.device("cpu")
    for model_name, model_file in config["models"].items():
        print(f"  Loading {model_name}...")

        # Verify file existence first
        full_path = os.path.join(utils.MODELS_DIR, model_file)
        if not os.path.exists(full_path):
            print(f"    Error: Model file not found at {full_path}")
            continue

        try:
            model, scaler, encoders = utils.load_model_and_scaler(
                model_name, name, device
            )
            if model:
                print(f"    Success: Model loaded.")
            else:
                print(f"    Failed: Model returned None.")

            if scaler:
                print(f"    Success: Scaler loaded.")
            else:
                print(f"    Failed: Scaler returned None.")

            if name == "UNSW-NB15":
                if encoders:
                    print(f"    Success: Encoders loaded.")
                else:
                    print(f"    Failed: Encoders returned None.")
        except Exception as e:
            print(f"    Error loading {model_name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    verify_dataset("NSL-KDD")
    verify_dataset("UNSW-NB15")
    verify_dataset("CICIDS2017")
