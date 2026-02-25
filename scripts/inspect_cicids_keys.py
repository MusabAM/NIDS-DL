import os

import torch


def inspect():
    path = "results/models/best_transformer_cicids2017.pth"
    if not os.path.exists(path):
        print("File not found")
        return

    print(f"Inspecting {os.path.basename(path)}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    for k in state_dict.keys():
        print(f"{k}: {state_dict[k].shape}")


if __name__ == "__main__":
    inspect()
