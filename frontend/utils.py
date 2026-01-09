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
        self.dense_block = nn.Sequential(
            nn.Linear(lstm_units[-1] * self.num_directions, dense_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units[0], dense_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.output_layer = nn.Linear(dense_units[1], num_classes)

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
# 2. Configuration & Paths
# ==============================================================================
MODELS_DIR = "results/models/"
SCALER_PATH = "results/models/cnn_scaler.pkl"
TRAIN_DATA_PATH = "data/raw/nsl-kdd/train.txt"

COLUMNS = [
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

# ==============================================================================
# 3. Helper Functions
# ==============================================================================


@st.cache_resource
def load_feature_columns():
    """
    Load training data to determine the full set of feature columns after one-hot encoding.
    This ensures consistent input shape for the model.
    """
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATA_PATH}. Needed for feature alignment."
        )

    df = pd.read_csv(TRAIN_DATA_PATH, header=None, names=COLUMNS)

    # Preprocessing
    df = df.drop("difficulty_level", axis=1)

    # One-hot encoding
    categorical_cols = ["protocol_type", "service", "flag"]
    encoded = pd.get_dummies(df, columns=categorical_cols)

    # Drop label
    drop_cols = ["label"]
    feature_cols = sorted([c for c in encoded.columns if c not in drop_cols])

    return feature_cols


@st.cache_resource
def load_model_and_scaler(model_name, device):
    """Load the trained model and scaler."""
    model_files = {
        "CNN": "cnn_nsl_kdd.pt",
        "LSTM": "best_lstm_kdd.pt",
        "Transformer": "transformer_nsl_kdd.pth",
    }

    if model_name not in model_files:
        return None, None

    path = os.path.join(MODELS_DIR, model_files[model_name])
    if not os.path.exists(path):
        return None, None

    # Load Scaler
    if not os.path.exists(SCALER_PATH):
        return None, None

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load Model structure and weights
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        return None, None

    # Handle state dict structure
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_dim = checkpoint.get("input_dim", 122)
    else:
        state_dict = checkpoint
        input_dim = 122

    if model_name == "CNN":
        model = CNNClassifier(input_dim=input_dim, num_classes=2)
    elif model_name == "LSTM":
        model = LSTMClassifier(input_dim=input_dim, num_classes=2)
    elif model_name == "Transformer":
        model = TransformerClassifier(input_dim=input_dim, num_classes=2)
    else:
        return None, None

    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None

    model.to(device)
    model.eval()

    return model, scaler


def preprocess_input(df, scaler, feature_cols):
    """
    Preprocess input dataframe to match model requirements:
    1. One-hot encode categorical variables.
    2. Add missing columns (filled with 0).
    3. Scale features.
    """
    # Ensure all categorical columns exist in input, even if empty, for get_dummies
    categorical_cols = ["protocol_type", "service", "flag"]

    # Encode
    encoded = pd.get_dummies(
        df, columns=[c for c in categorical_cols if c in df.columns]
    )

    # Add missing columns
    for col in feature_cols:
        if col not in encoded.columns:
            encoded[col] = 0

    # Reorder and select columns
    X = encoded[feature_cols].values

    # Scale
    X_scaled = scaler.transform(X)

    return X_scaled
