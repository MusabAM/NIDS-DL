import os
import sys

import torch

# Add project root to path
sys.path.append(os.getcwd())

from frontend.utils import TransformerClassifier, load_feature_columns


def check_transformer():
    print("Checking Transformer UNSW-NB15...")

    # Path
    model_path = "results/models/transformer_unsw.pt"
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    # Load State Dict
    try:
        # weights_only=False to allow scalers etc
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        print("State Dict Loaded.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Instantiate Model
    # "UNSW-NB15": { "embed_dim": 64, "num_heads": 4, "num_blocks": 3, "ff_dim": 128, "dense_units": [64], "dropout": 0.3 }

    # input_dim for UNSW is 196 (from feature_cols)
    input_dim = 196

    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=2,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dense_units=[64],
        dropout=0.3,
    )

    print("Model Instantiated.")

    # Try Load
    try:
        keys = model.load_state_dict(state_dict, strict=False)
        print(f"Load Result: {keys}")

        if len(keys.missing_keys) > 0:
            print("\nMissing Keys:")
            for k in keys.missing_keys:
                print(f"  {k}")

        if len(keys.unexpected_keys) > 0:
            print("\nUnexpected Keys:")
            for k in keys.unexpected_keys:
                print(f"  {k}")

    except RuntimeError as e:
        print(f"\nRuntimeError during load_state_dict: {e}")


if __name__ == "__main__":
    check_transformer()
