import os
import sys

import torch

try:
    path = "results/models/cnn_nsl_kdd.pt"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(0)

    checkpoint = torch.load(path, map_location="cpu")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    print("Model keys example:")
    for k in list(state_dict.keys())[:5]:
        print(k)

    if (
        "classifier.0.weight" in state_dict
        or "module.classifier.0.weight" in state_dict
    ):
        print("Detected: ImprovedCNNClassifier structure")
    elif (
        "dense_block.0.weight" in state_dict
        or "module.dense_block.0.weight" in state_dict
    ):
        print("Detected: CNNClassifier structure")
    elif "conv_block.0.weight" in state_dict:
        print("Detected: Generic CNN structure")

except Exception as e:
    print(f"Error: {e}")
