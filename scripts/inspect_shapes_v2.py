import os

import torch


def inspect_file(path, keys):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"\nInspecting {os.path.basename(path)}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        for k, v in state_dict.items():
            for key_frag in keys:
                if key_frag in k:
                    print(f"  {k}: {v.shape}")
    except Exception as e:
        print(f"  Error loading: {e}")


def main():
    models_dir = "results/models/"

    # UNSW Transformer PE check
    inspect_file(
        os.path.join(models_dir, "transformer_unsw.pt"), ["pos_encoding.pe", "pe"]
    )

    # CICIDS2017 Transformer
    inspect_file(
        os.path.join(models_dir, "best_transformer_cicids2017.pth"),
        ["encoder", "classifier", "weight"],
    )

    # CICIDS2017 Autoencoder
    inspect_file(
        os.path.join(models_dir, "best_autoencoder_cicids2017.pth"),
        ["encoder", "decoder", "weight"],
    )


if __name__ == "__main__":
    main()
