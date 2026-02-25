import os
import sys

import torch


def inspect_checkpoint(path):
    print(f"Inspecting {path}...")
    if not os.path.exists(path):
        print("  File not found.")
        return

    try:
        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("  Structure: dict with 'model_state_dict'")
            if "input_dim" in checkpoint:
                print(f"  Saved input_dim: {checkpoint['input_dim']}")
        else:
            state_dict = checkpoint
            print("  Structure: direct state_dict")

        print("  Keys and Shapes:")
        for k, v in state_dict.items():
            print(f"    {k}: {v.shape}")

    except Exception as e:
        print(f"  Error loading: {e}")


if __name__ == "__main__":
    models_dir = "results/models/"

    files = ["autoencoder_unsw.pt"]
    for f in files:
        inspect_checkpoint(os.path.join(models_dir, f))
        print("-" * 40)
