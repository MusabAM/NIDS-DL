import os

import torch


def inspect():
    path = "results/models/transformer_unsw.pt"
    if not os.path.exists(path):
        print("File not found")
        return

    print(f"Inspecting {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    keys_of_interest = [
        "input_embedding.weight",
        "pos_encoding.pe",
        "classification_head.0.weight",
        "output_layer.weight",
    ]

    for k, v in state_dict.items():
        if any(x in k for x in keys_of_interest) or "weight" in k:
            print(f"{k}: {v.shape}")


if __name__ == "__main__":
    inspect()
