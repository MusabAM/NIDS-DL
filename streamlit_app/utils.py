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


class CNNClassifierUNSW(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=2,
        conv_channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        dropout=0.3,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(1)

        layers = []
        in_channels = 1

        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)
        self._conv_output_size = self._get_conv_output_size(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self._conv_output_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def _get_conv_output_size(self, input_dim):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_dim)
            x = self.input_bn(x)
            x = self.conv_block(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.input_bn(x)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


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
            dense_layers.extend(
                [
                    nn.Linear(in_feat, units),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
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
    """
    Positional encoding for Transformer.
    Adds position information to the input embeddings.
    """

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model) or (seq_len, batch, d_model)
        # self.pe: (1, max_len, d_model)

        # If x is batch_first (batch, seq_len, d_model)
        if x.dim() == 3 and x.size(1) <= self.pe.size(1):
            x = x + self.pe[:, : x.size(1), :]
        # If x is seq_first (seq_len, batch, d_model)
        elif x.dim() == 3 and x.size(0) <= self.pe.size(1):
            # Transpose pe to (seq_len, 1, d_model) for broadcasting
            pe_sliced = self.pe[:, : x.size(0), :].transpose(0, 1)
            x = x + pe_sliced
        else:
            # Fallback or error for unexpected dimensions/sizes
            # For now, assume batch_first and hope for the best or raise error
            # print(f"Warning: PositionalEncoding received unexpected input shape {x.shape}. Applying default batch_first PE.")
            if x.dim() == 3 and x.size(1) <= self.pe.size(1):
                x = x + self.pe[:, : x.size(1), :]
            else:
                raise ValueError(
                    f"PositionalEncoding received input of shape {x.shape} which cannot be broadcast with PE of shape {self.pe.shape}"
                )

        return self.dropout(x)


# ==============================================================================
# UNSW Transformer Components
# ==============================================================================
class TransformerBlock(nn.Module):
    """Single transformer encoder block for UNSW model."""

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
        # x is (seq, batch, embed)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)


class TransformerClassifierUNSW(nn.Module):
    """
    Transformer-based classifier for UNSW-NB15 (Custom Architecture).
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
        num_blocks=4,
        dense_units=[128],
        dropout=0.1,
    ):
        super().__init__()
        self.input_embedding = nn.Linear(1, embed_dim)

        # UNSW Pos Encoding usually (max, 1, dim).
        # But we can reuse the global PositionalEncoding if we adapt.
        # Or define a local one. Let's reuse and adapt in forward.
        self.pos_encoding = PositionalEncoding(
            embed_dim, max_len=input_dim, dropout=dropout
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        head_layers = []
        in_features = embed_dim
        for units in dense_units:
            head_layers.extend(
                [
                    nn.Linear(in_features, units),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = units

        self.classification_head = nn.Sequential(*head_layers)
        self.output_layer = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        x = self.input_embedding(x)  # (batch, seq, embed)

        # Transpose to (seq, batch, embed)
        x = x.transpose(0, 1)

        # Add PE.
        # PE in utils is (1, max, dim). x is (seq, batch, dim).
        # We need PE to be (seq, 1, dim).
        # Let's transpose PE or slice.
        # self.pos_encoding.pe is (1, max, dim).
        # slice: pe[:, :seq, :] -> (1, seq, dim).
        # transpose: (seq, 1, dim).
        pe_slice = self.pos_encoding.pe[:, : x.size(0), :].transpose(0, 1)
        x = x + pe_slice
        x = self.pos_encoding.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        # Transpose back
        x = x.transpose(0, 1)  # (batch, seq, embed)
        x = x.mean(dim=1)

        x = self.classification_head(x)
        return self.output_layer(x)


class TransformerClassifierCICIDS(nn.Module):
    """
    Transformer-based classifier for CICIDS2017 (Flat Input).
    Treats the entire feature vector as a SINGLE token.
    """

    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        num_classes=2,
        dropout=0.3,
        dense_units=[64],
    ):
        super(TransformerClassifierCICIDS, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features (flat) to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (likely unused if seq_len=1, but might be present)
        # Checkpoint lacks it, so we can omit or optionalize.
        # Standard TransformerClassifier has it.
        # But if strict=False, we can keep it or not.
        # If we keep it, we need to ensure shape matches or it is ignored.
        # Let's keep it for compatibility sake but know it's useless for seq_len=1.
        self.pos_encoder = PositionalEncoding(d_model, max_len=10, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dense_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units[0], num_classes),
        )

    def forward(self, x):
        # x: (batch, input_dim)

        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch, d_model)

        # Reshape to (batch, 1, d_model) to act as sequence of length 1
        x = x.unsqueeze(1)

        # Add positional encoding (broadcasting works)
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, 1, d_model)

        # Global average pooling (trivial for seq_len=1)
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        x = self.classifier(x)  # (batch, num_classes)

        return x


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for NIDS.
    Treats each feature as a token in a sequence.
    """

    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        num_classes=2,
        dropout=0.3,
        dense_units=[64],
    ):
        super(TransformerClassifier, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=input_dim, dropout=dropout
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        # Use simple Sequential as in notebook (adapting to use dense_units arg for flexibility if needed,
        # but defaulting to notebook structure)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dense_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units[0], num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, 1) if input is (batch, input_dim) we need to unsqueeze
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        x = self.classifier(x)  # (batch, num_classes)

        return x


class Autoencoder(nn.Module):
    """
    Autoencoder for anomaly detection.
    """

    def __init__(
        self,
        input_dim,
        encoder_units=[64, 32, 16],
        latent_dim=8,
        decoder_units=None,
        dropout_rate=0.2,
        activation="relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if decoder_units is None:
            decoder_units = list(reversed(encoder_units))

        # Activation function
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        self.activation = activations.get(activation, nn.ReLU())

        # Build encoder
        encoder_layers = []
        in_features = input_dim

        for units in encoder_units:
            encoder_layers.extend(
                [
                    nn.Linear(in_features, units),
                    nn.BatchNorm1d(units),
                    self.activation,
                    nn.Dropout(dropout_rate),
                ]
            )
            in_features = units

        encoder_layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        in_features = latent_dim

        for units in decoder_units:
            decoder_layers.extend(
                [
                    nn.Linear(in_features, units),
                    nn.BatchNorm1d(units),
                    self.activation,
                    nn.Dropout(dropout_rate),
                ]
            )
            in_features = units

        decoder_layers.append(nn.Linear(in_features, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        self.threshold = None

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def reconstruction_error(self, x):
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            mse = (x_recon - x) ** 2
            return mse.mean(dim=1)


# ==============================================================================
# 2. Dataset Configuration
# ==============================================================================
# Get the project root directory (parent of frontend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
SCALER_PATH = os.path.join(MODELS_DIR, "cnn_scaler.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "nsl-kdd", "train.txt")

# --- NSL-KDD Configuration ---
NSL_KDD_CONFIG = {
    "scaler_path": os.path.join(MODELS_DIR, "cnn_scaler.pkl"),
    "train_data_path": TRAIN_DATA_PATH,
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
    "scaler_path": os.path.join(MODELS_DIR, "cicids2018_scaler.pkl"),
    "feature_cols_path": os.path.join(MODELS_DIR, "cicids2018_feature_cols.pkl"),
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
    "scaler_path": os.path.join(MODELS_DIR, "cicids2017_scaler.pkl"),
    "feature_cols_path": os.path.join(MODELS_DIR, "cicids2017_feature_cols.pkl"),
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

# Setup UNSW-NB15 config
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
DATASET_CONFIG = DATASET_CONFIGS

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


@st.cache_data
def load_unsw_nb15_feature_columns():
    feature_cols_path = DATASET_CONFIGS["UNSW-NB15"]["feature_cols_path"]
    if not os.path.exists(feature_cols_path):
        return None
    with open(feature_cols_path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(dataset="NSL-KDD"):
    """Load feature columns for the specified dataset."""
    if dataset == "NSL-KDD":
        return load_nsl_kdd_feature_columns()
    elif dataset == "UNSW-NB15":
        return load_unsw_nb15_feature_columns()
    elif dataset == "UNSW-NB15":
        return preprocess_unsw_nb15_input(df, scaler, feature_cols, encoders)
    elif dataset == "CICIDS2018":
        return load_cicids2018_feature_columns()
    elif dataset == "CICIDS2017":
        return load_cicids2017_feature_columns()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


@st.cache_resource
def load_model_and_scaler(model_name, dataset, device):
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

    encoders = None
    if "encoders_path" in config:
        if os.path.exists(config["encoders_path"]):
            with open(config["encoders_path"], "rb") as f:
                encoders = pickle.load(f)

    # Load model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None, None, None

    # Handle state dict structure
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_dim = checkpoint.get("input_dim", scaler.n_features_in_)
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
        input_dim = (
            scaler.n_features_in_
            if hasattr(scaler, "n_features_in_")
            else checkpoint.get(
                "input_dim",
                (
                    len(load_feature_columns(dataset))
                    if load_feature_columns(dataset)
                    else 41
                ),
            )
        )

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
        model = Autoencoder(
            input_dim=input_dim, encoder_units=encoder_units, latent_dim=latent_dim
        )
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

    keys_to_remove = [
        k for k in new_state_dict.keys() if "pos_encoding.pe" in k or ".pe" in k
    ]
    for k in keys_to_remove:
        del new_state_dict[k]
    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        st.error(f"Error loading model weights for {model_name}: {e}")
        return None, None, None

    model.to(device)
    model.eval()

    return model, scaler, encoders


def preprocess_unsw_nb15_input(df, scaler, feature_cols, encoders=None):
    if encoders:
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")
                df[col] = le.transform(df[col])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].values
    return scaler.transform(X)


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


def preprocess_input(df, scaler, feature_cols, encoders=None, dataset="NSL-KDD"):
    """Preprocess input dataframe based on dataset type."""
    if dataset == "NSL-KDD":
        return preprocess_nsl_kdd_input(df, scaler, feature_cols)
    elif dataset == "UNSW-NB15":
        return load_unsw_nb15_feature_columns()
    elif dataset == "UNSW-NB15":
        return preprocess_unsw_nb15_input(df, scaler, feature_cols, encoders)
    elif dataset == "CICIDS2018":
        return preprocess_cicids2018_input(df, scaler, feature_cols)
    elif dataset == "CICIDS2017":
        return preprocess_cicids2017_input(df, scaler, feature_cols)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
