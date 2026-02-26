import os
import re


def resolve_utils_py():
    with open(
        "c:/Users/musab/Projects/NIDS-DL/frontend/utils.py", "r", encoding="utf-8"
    ) as f:
        content = f.read()

    # Block 1 - Base Dir
    # HEAD has robust paths, INCOMING deleted them. We keep HEAD's robust paths, but we remove the merge markers.
    content = re.sub(
        r"<<<<<<< HEAD\n# Get the project root directory(?:.|\n)*?# --- NSL-KDD Configuration ---",
        """# Get the project root directory (parent of frontend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
SCALER_PATH = os.path.join(MODELS_DIR, "cnn_scaler.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "nsl-kdd", "train.txt")

# --- NSL-KDD Configuration ---""",
        content,
    )

    # Block 2 - DATASET_CONFIG dictionary
    # We want to replace HEAD's DATASET_CONFIG definition and INCOMING's COLUMNS = ... backward compatibility
    # The conflict marker is around line 718-755:
    content = re.sub(
        r"<<<<<<< HEAD\n# Dataset Specific Configs\nDATASET_CONFIG = \{(?:.|\n)*?\}\n=======\n# Keep backward compatibility\nCOLUMNS = NSL_KDD_COLUMNS\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f",
        """# Setup UNSW-NB15 config
UNSW_NB15_CONFIG = {
    "scaler_path": os.path.join(MODELS_DIR, "unsw_scaler.pkl"),
    "encoders_path": os.path.join(MODELS_DIR, "unsw_encoders.pkl"),
    "feature_cols_path": os.path.join(MODELS_DIR, "unsw_feature_cols.pkl"),
    "model_files": {
        "CNN": "cnn_unsw_nb15.pt",
        "LSTM": "best_lstm_unsw.pt",
        "Transformer": "transformer_unsw.pt",
        "Autoencoder": "autoencoder_unsw.pt",
    },
    "model_params": {
        "CNN": {"num_classes": 2},
        "LSTM": {
            "num_classes": 2,
            "lstm_units": [128, 64],
            "dense_units": [128, 64],
            "bidirectional": True,
            "dropout_rate": 0.3,
        },
        "Transformer": {
            "num_classes": 2,
            "embed_dim": 64,
            "num_heads": 4,
            "ff_dim": 128,
            "num_blocks": 3,
            "dense_units": [64],
            "dropout": 0.3,
        },
    },
}

DATASET_CONFIGS["UNSW-NB15"] = UNSW_NB15_CONFIG

# Keep backward compatibility
COLUMNS = NSL_KDD_COLUMNS
DATASET_CONFIG = DATASET_CONFIGS""",
        content,
    )

    # We need to make sure INCOMING's absolute paths in DATASET_CONFIGS are made relative to MODELS_DIR
    content = content.replace(
        '"results/models/cnn_scaler.pkl"', 'os.path.join(MODELS_DIR, "cnn_scaler.pkl")'
    )
    content = content.replace(
        '"results/models/cicids2018_scaler.pkl"',
        'os.path.join(MODELS_DIR, "cicids2018_scaler.pkl")',
    )
    content = content.replace(
        '"results/models/cicids2018_feature_cols.pkl"',
        'os.path.join(MODELS_DIR, "cicids2018_feature_cols.pkl")',
    )
    content = content.replace(
        '"results/models/cicids2017_scaler.pkl"',
        'os.path.join(MODELS_DIR, "cicids2017_scaler.pkl")',
    )
    content = content.replace(
        '"results/models/cicids2017_feature_cols.pkl"',
        'os.path.join(MODELS_DIR, "cicids2017_feature_cols.pkl")',
    )
    content = content.replace('"data/raw/nsl-kdd/train.txt"', "TRAIN_DATA_PATH")

    # Block 3: load_feature_columns
    content = re.sub(
        r'<<<<<<< HEAD\n@st\.cache_resource\ndef load_feature_columns\(dataset_name="NSL-KDD"\):(.*?)\n=======\n@st\.cache_data\ndef load_nsl_kdd_feature_columns\(\):\n',
        "@st.cache_data\ndef load_nsl_kdd_feature_columns():\n",
        content,
        flags=re.DOTALL,
    )

    # We need to add load_unsw_nb15_feature_columns into utils.py
    # Let's insert it after load_cicids2017_feature_columns
    unsw_func = """
@st.cache_data
def load_unsw_nb15_feature_columns():
    feature_cols_path = DATASET_CONFIGS["UNSW-NB15"]["feature_cols_path"]
    if not os.path.exists(feature_cols_path):
        return None
    with open(feature_cols_path, "rb") as f:
        return pickle.load(f)
"""
    content = content.replace(
        'def load_feature_columns(dataset="NSL-KDD"):',
        unsw_func + '\ndef load_feature_columns(dataset="NSL-KDD"):',
    )

    # Update load_feature_columns dispatch
    content = content.replace(
        'elif dataset == "CICIDS2018":',
        'elif dataset == "UNSW-NB15":\n        return load_unsw_nb15_feature_columns()\n    elif dataset == "CICIDS2018":',
    )

    # Block 4: load_model_and_scaler
    # The incoming function signature is load_model_and_scaler(model_name, device, dataset="NSL-KDD") ... wait, in app.py it said New Signature: load_model_and_scaler(model_name, dataset, device).
    # Oh! INCOMING's utils.py had `load_model_and_scaler(model_name, device, dataset="NSL-KDD")`
    # Let's clean up HEAD's load_model_and_scaler and just use INCOMING's but adapt it.
    to_replace_model = r'<<<<<<< HEAD\ndef load_model_and_scaler\(model_name, dataset_name, device\):(.*?)\n=======\ndef load_model_and_scaler\(model_name, device, dataset="NSL-KDD"\):(.*?)\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f'
    # Actually, the conflict blocks are split into 3-4 chunks in this function.
    # Let's handle the entire function `load_model_and_scaler`!

    # Wait, it's easier to just parse it line by line or find the chunks.
    # Chunk 4a:
    content = re.sub(
        r'<<<<<<< HEAD\ndef load_model_and_scaler\(model_name, dataset_name, device\):(.*?)\n=======\ndef load_model_and_scaler\(model_name, device, dataset="NSL-KDD"\):(.*?)\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f',
        r"def load_model_and_scaler(model_name, dataset, device):\n\2",
        content,
        flags=re.DOTALL,
    )

    # Chunk 4b: checkpoint loading
    content = re.sub(
        r'<<<<<<< HEAD\n        # Load checkpoint relative to device(.*?)\n=======\n        checkpoint = torch\.load\(model_path, map_location=device, weights_only=False\)\n    except Exception as e:\n        st\.error\(f"Error loading model checkpoint: \{e\}"\)\n        return None, None\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f',
        r'        checkpoint = torch.load(model_path, map_location=device, weights_only=False)\n    except Exception as e:\n        st.error(f"Error loading model checkpoint: {e}")\n        return None, None, None',
        content,
        flags=re.DOTALL,
    )

    # Chunk 4c: model instantiation
    content = re.sub(
        r'<<<<<<< HEAD\n        input_dim = checkpoint\.get\("input_dim"\)(.*?)\n=======\n        input_dim = checkpoint\.get\("input_dim", scaler\.n_features_in_\)\n    else:\n        state_dict = checkpoint if isinstance\(checkpoint, dict\) else checkpoint\n        input_dim = scaler\.n_features_in_\n\n    # Get model-specific params from config\n    params = config\["model_params"\].get\(model_name, \{\}\)\n\n    # Create model with CORRECT architecture for this dataset\n    if model_name == "CNN":\n        model = CNNClassifier\(input_dim=input_dim, \*\*params\)\n    elif model_name == "LSTM":\n        model = LSTMClassifier\(input_dim=input_dim, \*\*params\)\n    elif model_name == "Transformer":\n        model = TransformerClassifier\(input_dim=input_dim, \*\*params\)\n    else:\n        return None, None\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f',
        """        input_dim = checkpoint.get("input_dim", scaler.n_features_in_)
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
        input_dim = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else checkpoint.get("input_dim", len(load_feature_columns(dataset)) if load_feature_columns(dataset) else 41)

    params = config["model_params"].get(model_name, {})

    if model_name == "CNN":
        if dataset == "UNSW-NB15":
            model = CNNClassifierUNSW(input_dim=input_dim, num_classes=2)
        else:
            model = CNNClassifier(input_dim=input_dim, **params)
    elif model_name == "LSTM":
        model = LSTMClassifier(input_dim=input_dim, **params)
    elif model_name == "Transformer":
        if dataset == "UNSW-NB15":
            model = TransformerClassifierUNSW(input_dim=input_dim, **params)
        elif dataset == "CICIDS2017":
            model = TransformerClassifierCICIDS(input_dim=input_dim, **params)
        else:
            model = TransformerClassifier(input_dim=input_dim, **params)
    elif model_name == "Autoencoder":
        encoder_units = [64, 32, 16]
        latent_dim = 8
        if dataset == "UNSW-NB15":
            encoder_units = [256, 128, 64]
            latent_dim = 32
        elif dataset == "CICIDS2017":
            encoder_units = [128, 64]
            latent_dim = 32
        model = Autoencoder(input_dim=input_dim, encoder_units=encoder_units, latent_dim=latent_dim)
    else:
        return None, None, None

    # Handle state dict changes
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("fc."):
            new_key = k.replace("fc.", "classifier.")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    keys_to_remove = [k for k in new_state_dict.keys() if "pos_encoding.pe" in k or ".pe" in k]
    for k in keys_to_remove: del new_state_dict[k]
    state_dict = new_state_dict""",
        content,
        flags=re.DOTALL,
    )

    # We need to make sure INCOMING's `return None, None` becomes `return None, None, None` and returning `encoders` works!
    # In `load_model_and_scaler`, `encoders` loading must be patched:
    enc_patch = """
    encoders = None
    if "encoders_path" in config:
        if os.path.exists(config["encoders_path"]):
            with open(config["encoders_path"], "rb") as f:
                encoders = pickle.load(f)
    """
    content = content.replace(
        'with open(scaler_path, "rb") as f:\n        scaler = pickle.load(f)',
        'with open(scaler_path, "rb") as f:\n        scaler = pickle.load(f)\n'
        + enc_patch,
    )

    # Change the return at the end of load_model_and_scaler
    content = content.replace(
        "return model, scaler\n", "return model, scaler, encoders\n"
    )

    # Block 5: preprocess_input
    content = re.sub(
        r"<<<<<<< HEAD\ndef preprocess_input(.*?)=======\ndef preprocess_nsl_kdd_input",
        'def preprocess_unsw_nb15_input(df, scaler, feature_cols, encoders=None):\n    if encoders:\n        for col, le in encoders.items():\n            if col in df.columns:\n                df[col] = df[col].fillna("unknown").astype(str)\n                df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")\n                df[col] = le.transform(df[col])\n    for col in feature_cols:\n        if col not in df.columns: df[col] = 0\n    X = df[feature_cols].values\n    return scaler.transform(X)\n\ndef preprocess_nsl_kdd_input',
        content,
        flags=re.DOTALL,
    )

    content = re.sub(
        r">>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f\n    X_scaled = scaler\.transform\(X\)",
        "    X_scaled = scaler.transform(X)",
        content,
    )

    # Adjust main preprocess_input dispatch
    content = content.replace(
        'def preprocess_input(df, scaler, feature_cols, dataset="NSL-KDD"):',
        'def preprocess_input(df, scaler, feature_cols, encoders=None, dataset="NSL-KDD"):',
    )
    content = content.replace(
        'elif dataset == "CICIDS2018":',
        'elif dataset == "UNSW-NB15":\n        return preprocess_unsw_nb15_input(df, scaler, feature_cols, encoders)\n    elif dataset == "CICIDS2018":',
    )

    with open(
        "c:/Users/musab/Projects/NIDS-DL/frontend/utils.py", "w", encoding="utf-8"
    ) as f:
        f.write(content)


resolve_utils_py()
