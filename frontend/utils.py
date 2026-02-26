import math
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ==============================================================================
# 1. Custom Model Architectures
# ==============================================================================


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout_rate=0.3):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256 * 8, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (B, F) -> (B, 1, F)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.fc3(x)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=2,
        lstm_units=[128, 64],
        dense_units=[128, 64],
        bidirectional=True,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm_layers = nn.ModuleList()
        in_features = input_dim
        for i, units in enumerate(lstm_units):
            lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=units,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.lstm_layers.append(lstm)
            in_features = units * self.num_directions
        self.lstm_dropout = nn.Dropout(dropout_rate)

        # Build dense block dynamically for any number of dense_units
        dense_layers = []
        in_feat = lstm_units[-1] * self.num_directions
        for units in dense_units:
            dense_layers.extend([
                nn.Linear(in_feat, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_feat = units
        self.dense_block = nn.Sequential(*dense_layers)
        self.output_layer = nn.Linear(dense_units[-1], num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = x[:, -1, :]
        x = self.lstm_dropout(x)
        x = self.dense_block(x)
        return self.output_layer(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=2,
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        num_blocks=4,
        dense_units=[128],
        dropout=0.1,
    ):
        super().__init__()
        self.input_embedding = nn.Linear(1, embed_dim)
        self.pos_encoding = PositionalEncoding(
            embed_dim, max_len=input_dim, dropout=dropout
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, dense_units[0]), nn.ReLU(), nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(dense_units[0], num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embedding(x).transpose(0, 1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.transpose(0, 1).mean(dim=1)
        x = self.classification_head(x)
        return self.output_layer(x)


# ==============================================================================
# 2. Dataset Configuration
# ==============================================================================

MODELS_DIR = "results/models/"

# --- NSL-KDD Configuration ---
NSL_KDD_CONFIG = {
    "scaler_path": "results/models/cnn_scaler.pkl",
    "train_data_path": "data/raw/nsl-kdd/train.txt",
    "model_files": {
        "CNN": "cnn_nsl_kdd.pt",
        "LSTM": "best_lstm_kdd.pt",
        "Transformer": "transformer_nsl_kdd.pth",
    },
    # Architecture params matching NSL-KDD training
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
            "embed_dim": 128,
            "num_heads": 8,
            "ff_dim": 256,
            "num_blocks": 4,
            "dense_units": [128],
            "dropout": 0.1,
        },
    },
}

# --- CICIDS2018 Configuration ---
CICIDS2018_CONFIG = {
    "scaler_path": "results/models/cicids2018_scaler.pkl",
    "feature_cols_path": "results/models/cicids2018_feature_cols.pkl",
    "model_files": {
        "CNN": "best_cnn_cicids2018.pth",
        "LSTM": "best_lstm_cicids2018.pth",
        "Transformer": "transformer_cicids2018_best.pt",
    },
    # Architecture params matching CICIDS2018 training notebooks exactly
    "model_params": {
        "CNN": {"num_classes": 2, "dropout_rate": 0.3},
        "LSTM": {
            "num_classes": 2,
            "lstm_units": [128, 64],
            "dense_units": [64],  # Training used [64], NOT [128, 64]
            "bidirectional": True,
            "dropout_rate": 0.3,
        },
        "Transformer": {
            "num_classes": 2,
            "embed_dim": 64,  # Training used 64, NOT 128
            "num_heads": 4,  # Training used 4, NOT 8
            "ff_dim": 128,  # Training used 128, NOT 256
            "num_blocks": 3,  # Training used 3, NOT 4
            "dense_units": [64],  # Training used [64], NOT [128]
            "dropout": 0.3,
        },
    },
}

# --- CICIDS2017 Configuration ---
CICIDS2017_CONFIG = {
    "scaler_path": "results/models/cicids2017_scaler.pkl",
    "feature_cols_path": "results/models/cicids2017_feature_cols.pkl",
    "model_files": {
        "LSTM": "best_lstm_cicids2017.pth",
        "Transformer": "best_transformer_cicids2017.pth",
    },
    # Architecture params matching our improved versions
    "model_params": {
        "LSTM": {
            "num_classes": 2,
            "lstm_units": [128, 64],
            "dense_units": [128, 64],
            "bidirectional": True,
            "dropout_rate": 0.3,
        },
        "Transformer": {
            "num_classes": 2,
            "embed_dim": 128,
            "num_heads": 8,
            "ff_dim": 256,
            "num_blocks": 4,
            "dense_units": [128],
            "dropout": 0.1,
        },
    },
}

DATASET_CONFIGS = {
    "NSL-KDD": NSL_KDD_CONFIG,
    "CICIDS2018": CICIDS2018_CONFIG,
    "CICIDS2017": CICIDS2017_CONFIG,
}

# Metadata columns to drop from CICIDS2018 data
CICIDS2018_DROP_COLS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
]

# Metadata columns to drop from CICIDS2017 data
CICIDS2017_DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Label",
]

# NSL-KDD column names (original 41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

# Keep backward compatibility
COLUMNS = NSL_KDD_COLUMNS

# ==============================================================================
# 3. Helper Functions
# ==============================================================================


@st.cache_data
def load_nsl_kdd_feature_columns():
    """
    Load NSL-KDD training data to determine the full set of feature columns
    after one-hot encoding.
    """
    train_data_path = NSL_KDD_CONFIG["train_data_path"]
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Training data not found at {train_data_path}. Needed for feature alignment."
        )

    df = pd.read_csv(train_data_path, header=None, names=NSL_KDD_COLUMNS)
    df = df.drop("difficulty_level", axis=1)

    categorical_cols = ["protocol_type", "service", "flag"]
    encoded = pd.get_dummies(df, columns=categorical_cols)

    drop_cols = ["label"]
    feature_cols = sorted([c for c in encoded.columns if c not in drop_cols])

    return feature_cols


@st.cache_data
def load_cicids2018_feature_columns():
    """Load CICIDS2018 feature column names from saved pickle."""
    feature_cols_path = CICIDS2018_CONFIG["feature_cols_path"]
    if not os.path.exists(feature_cols_path):
        # Fallback: return None and let preprocessing handle it dynamically
        return None

    with open(feature_cols_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_cicids2017_feature_columns():
    """Load CICIDS2017 feature column names from saved pickle."""
    feature_cols_path = CICIDS2017_CONFIG["feature_cols_path"]
    if not os.path.exists(feature_cols_path):
        return None
    with open(feature_cols_path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(dataset="NSL-KDD"):
    """Load feature columns for the specified dataset."""
    if dataset == "NSL-KDD":
        return load_nsl_kdd_feature_columns()
    elif dataset == "CICIDS2018":
        return load_cicids2018_feature_columns()
    elif dataset == "CICIDS2017":
        return load_cicids2017_feature_columns()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


@st.cache_resource
def load_model_and_scaler(model_name, device, dataset="NSL-KDD"):
    """Load the trained model and scaler for the specified dataset."""
    config = DATASET_CONFIGS.get(dataset)
    if config is None:
        return None, None

    if model_name not in config["model_files"]:
        return None, None

    # Load model file
    model_path = os.path.join(MODELS_DIR, config["model_files"][model_name])
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None

    # Load scaler
    scaler_path = config["scaler_path"]
    if not os.path.exists(scaler_path):
        st.error(
            f"Scaler not found: {scaler_path}. "
            f"Run 'python scripts/generate_cicids2018_scaler.py' first."
            if dataset == "CICIDS2018"
            else f"Scaler not found: {scaler_path}"
        )
        return None, None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None, None

    # Handle state dict structure
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_dim = checkpoint.get("input_dim", scaler.n_features_in_)
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
        input_dim = scaler.n_features_in_

    # Get model-specific params from config
    params = config["model_params"].get(model_name, {})

    # Create model with CORRECT architecture for this dataset
    if model_name == "CNN":
        model = CNNClassifier(input_dim=input_dim, **params)
    elif model_name == "LSTM":
        model = LSTMClassifier(input_dim=input_dim, **params)
    elif model_name == "Transformer":
        model = TransformerClassifier(input_dim=input_dim, **params)
    else:
        return None, None

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None

    model.to(device)
    model.eval()

    return model, scaler


def preprocess_nsl_kdd_input(df, scaler, feature_cols):
    """
    Preprocess NSL-KDD input dataframe:
    1. One-hot encode categorical variables.
    2. Add missing columns (filled with 0).
    3. Scale features.
    """
    categorical_cols = ["protocol_type", "service", "flag"]

    encoded = pd.get_dummies(
        df, columns=[c for c in categorical_cols if c in df.columns]
    )

    for col in feature_cols:
        if col not in encoded.columns:
            encoded[col] = 0

    X = encoded[feature_cols].values
    X_scaled = scaler.transform(X)

    return X_scaled


def preprocess_cicids2018_input(df, scaler, feature_cols=None):
    """
    Preprocess CICIDS2018 input dataframe:
    1. Drop metadata columns.
    2. Convert to numeric.
    3. Handle NaN/Inf.
    4. Align columns with training features.
    5. Scale features.
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop metadata columns
    for col in CICIDS2018_DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col], errors="ignore")

    # Drop label-like columns
    for col in ["binary_label", "label", "Label"]:
        if col in df.columns:
            df = df.drop(columns=[col], errors="ignore")

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle Inf/NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)  # Fill NaN with 0 for inference (don't drop rows)

    # Align columns if feature_cols is available
    if feature_cols is not None:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_cols].values
    else:
        X = df.values

    X = X.astype(np.float32)
    X_scaled = scaler.transform(X)

    return X_scaled


def preprocess_cicids2017_input(df, scaler, feature_cols=None):
    """
    Preprocess CICIDS2017 input dataframe:
    1. Drop metadata columns.
    2. Convert to numeric.
    3. Handle NaN/Inf.
    4. Align columns.
    5. Scale features.
    """
    df.columns = df.columns.str.strip()
    for col in CICIDS2017_DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col], errors="ignore")
    for col in ["binary_label", "label", "Label"]:
        if col in df.columns:
            df = df.drop(columns=[col], errors="ignore")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    if feature_cols is not None:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_cols].values
    else:
        X = df.values
    X = X.astype(np.float32)
    X_scaled = scaler.transform(X)
    return X_scaled


def preprocess_input(df, scaler, feature_cols, dataset="NSL-KDD"):
    """Preprocess input dataframe based on dataset type."""
    if dataset == "NSL-KDD":
        return preprocess_nsl_kdd_input(df, scaler, feature_cols)
    elif dataset == "CICIDS2018":
        return preprocess_cicids2018_input(df, scaler, feature_cols)
    elif dataset == "CICIDS2017":
        return preprocess_cicids2017_input(df, scaler, feature_cols)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
