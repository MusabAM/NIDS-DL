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
# 2. Configuration & Paths
# ==============================================================================
# Get the project root directory (parent of frontend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
SCALER_PATH = os.path.join(MODELS_DIR, "cnn_scaler.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "nsl-kdd", "train.txt")

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

# Dataset Specific Configs
DATASET_CONFIG = {
    "NSL-KDD": {
        "models": {
            "CNN": "cnn_nsl_kdd.pt",
            "LSTM": "best_lstm_kdd.pt",
            "Transformer": "transformer_nsl_kdd.pth",
            "Autoencoder": "autoencoder_nsl_kdd.pt",
        },
        "scaler": "cnn_scaler.pkl",
        "feature_cols_file": None,  # Generated dynamically
    },
    "UNSW-NB15": {
        "models": {
            "CNN": "cnn_unsw_nb15.pt",
            "LSTM": "best_lstm_unsw.pt",
            "Transformer": "transformer_unsw.pt",
            "Autoencoder": "autoencoder_unsw.pt",
        },
        "scaler": "unsw_scaler.pkl",
        "encoders": "unsw_encoders.pkl",
        "feature_cols_file": "unsw_feature_cols.pkl",
    },
    "CICIDS2017": {
        "models": {
            "CNN": "best_cnn_cicids2017.pth",
            "LSTM": "best_lstm_cicids2017.pth",
            "Transformer": "best_transformer_cicids2017.pth",
            "Autoencoder": "best_autoencoder_cicids2017.pth",
        },
        "scaler": "cicids2017_scaler.pkl",
        "feature_cols_file": "cicids2017_feature_cols.pkl",
    },
}

# ==============================================================================
# 3. Helper Functions
# ==============================================================================


@st.cache_resource
def load_feature_columns(dataset_name="NSL-KDD"):
    """
    Load feature columns for the specified dataset.
    """
    if dataset_name == "NSL-KDD":
        if not os.path.exists(TRAIN_DATA_PATH):
            raise FileNotFoundError(
                f"Training data not found at {TRAIN_DATA_PATH}. Needed for feature alignment."
            )
        df = pd.read_csv(TRAIN_DATA_PATH, header=None, names=COLUMNS)
        df = df.drop("difficulty_level", axis=1)
        categorical_cols = ["protocol_type", "service", "flag"]
        encoded = pd.get_dummies(df, columns=categorical_cols)
        drop_cols = ["label"]
        feature_cols = sorted([c for c in encoded.columns if c not in drop_cols])
        return feature_cols

    elif dataset_name in ["UNSW-NB15", "CICIDS2017"]:
        fname = DATASET_CONFIG[dataset_name]["feature_cols_file"]
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return []

    return []


@st.cache_resource
def load_model_and_scaler(model_name, dataset_name, device):
    """Load the trained model and scaler for the specified dataset."""

    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        return None, None, None

    model_file = config["models"].get(model_name)
    if not model_file:
        return None, None, None

    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None, None

    # Load Scaler
    scaler_path = os.path.join(MODELS_DIR, config["scaler"])
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Load Encoders (if applicable)
    encoders = None
    if "encoders" in config:
        enc_path = os.path.join(MODELS_DIR, config["encoders"])
        if os.path.exists(enc_path):
            with open(enc_path, "rb") as f:
                encoders = pickle.load(f)

    # Load Model structure and weights
    try:
        # Load checkpoint relative to device
        # Set weights_only=False to allow loading checkpoints that might contain other objects (like scalers)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Error loading checkpoint {model_file}: {e}")
        return None, None, None

    # Handle state dict structure
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_dim = checkpoint.get("input_dim")
    else:
        state_dict = checkpoint
        input_dim = None

    # If input_dim not in checkpoint, infer from feature cols
    if input_dim is None:
        feature_cols = load_feature_columns(dataset_name)
        input_dim = len(feature_cols)

    # Transformer Configurations
    transformer_configs = {
        "NSL-KDD": {
            "embed_dim": 128,
            "num_heads": 4,
            "num_blocks": 4,
            "ff_dim": 256,
            "dense_units": [64],
            "dropout": 0.3,
        },
        "UNSW-NB15": {
            "embed_dim": 64,
            "num_heads": 4,
            "num_blocks": 3,
            "ff_dim": 128,
            "dense_units": [64],
            "dropout": 0.3,
        },
        "CICIDS2017": {
            "embed_dim": 128,
            "num_heads": 4,
            "num_blocks": 4,
            "ff_dim": 512,
            "dense_units": [64],
            "dropout": 0.3,
        },
    }

    # Model instantiation
    model = None
    if model_name == "CNN":
        if dataset_name == "UNSW-NB15":
            model = CNNClassifierUNSW(input_dim=input_dim, num_classes=2)
        else:
            model = CNNClassifier(input_dim=input_dim, num_classes=2)

    elif model_name == "LSTM":
        model = LSTMClassifier(input_dim=input_dim, num_classes=2)

    elif model_name == "Transformer":
        tf_config = transformer_configs.get(
            dataset_name, transformer_configs["NSL-KDD"]
        )

        if dataset_name == "UNSW-NB15":
            # UNSW trained on 42 features (Label Encoded)
            input_dim = 42
            model = TransformerClassifierUNSW(
                input_dim=input_dim,
                num_classes=2,
                embed_dim=tf_config["embed_dim"],
                num_heads=tf_config["num_heads"],
                ff_dim=tf_config["ff_dim"],
                num_blocks=tf_config["num_blocks"],
                dense_units=tf_config["dense_units"],
                dropout=tf_config["dropout"],
            )

        elif dataset_name == "CICIDS2017":
            # CICIDS2017 uses Flat Projection (Linear(77, 128))
            model = TransformerClassifierCICIDS(
                input_dim=input_dim,
                d_model=tf_config["embed_dim"],
                nhead=tf_config["num_heads"],
                num_layers=tf_config["num_blocks"],
                dim_feedforward=tf_config["ff_dim"],
                dense_units=tf_config["dense_units"],
                num_classes=2,
                dropout=tf_config["dropout"],
            )
        else:
            # NSL-KDD (Standard Sequence)
            model = TransformerClassifier(
                input_dim=input_dim,
                num_classes=2,
                d_model=tf_config["embed_dim"],
                nhead=tf_config["num_heads"],
                dim_feedforward=tf_config["ff_dim"],
                num_layers=tf_config["num_blocks"],
                dense_units=tf_config["dense_units"],
                dropout=tf_config["dropout"],
            )

    elif model_name == "Autoencoder":
        # Check input_dim consistency!
        # State dict usually has 'encoder.0.weight' shape [64, input_dim].
        # We can infer input_dim from state_dict if possible.
        if "encoder.0.weight" in state_dict:
            saved_input_dim = state_dict["encoder.0.weight"].shape[1]
            if saved_input_dim != input_dim:
                print(
                    f"DEBUG: Adjusting Autoencoder input_dim from {input_dim} to {saved_input_dim}"
                )
                input_dim = saved_input_dim

        # Specific configs for Autoencoder
        encoder_units = [64, 32, 16]  # Default
        latent_dim = 8

        if dataset_name == "UNSW-NB15":
            # Based on checkpoint inspection: Input 42 -> 256 -> 128 -> 64 -> 32(latent)
            encoder_units = [256, 128, 64]
            latent_dim = 32

        elif dataset_name == "CICIDS2017":
            # Based on checkpoint inspection: Input 77 -> 128 -> 64 -> 32(latent)
            encoder_units = [128, 64]
            latent_dim = 32
            # Also ensure input_dim matches
            if input_dim != 77 and "encoder.0.weight" in state_dict:
                saved_dim = state_dict["encoder.0.weight"].shape[1]
                if saved_dim != input_dim:
                    input_dim = saved_dim

        model = Autoencoder(
            input_dim=input_dim, encoder_units=encoder_units, latent_dim=latent_dim
        )

    if model is None:
        st.error(f"Model class for {model_name} not found.")
        return None, None, None

    # Fix State Dict Keys (e.g. fc -> classifier)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("fc."):
            new_key = k.replace("fc.", "classifier.")
            new_state_dict[new_key] = v
        # Also handle potential mismatch with 'classification_head' if previous attempts saved it
        elif k.startswith("classification_head."):
            # Map old classification_head + output_layer to classifier if needed,
            # but simpler to just rename prefix if it's a direct mapping.
            # Current TransformerClassifier uses 'classifier'.
            # If checkpoint has 'classification_head', it implies split.
            # Let's assume 'fc' is the main issue.
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    # Prevent shape mismatch for PositionalEncoding buffer (pe)
    # Since PE is fixed (sin/cos), we can safely skip loading it and rely on re-initialization
    keys_to_remove = [
        k for k in new_state_dict.keys() if "pos_encoding.pe" in k or ".pe" in k
    ]
    for k in keys_to_remove:
        # Check if shape mismatch exists
        # Actually easier to just valid since it is a buffer.
        # But let's only remove if strictly necessary?
        # Safest is to remove it always if specific key exists, as strict=False allows missing.
        del new_state_dict[k]

    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading model weights for {model_name}: {e}")
        return None, None, None

    model.to(device)
    model.eval()

    return model, scaler, encoders


def preprocess_input(df, scaler, feature_cols, encoders=None, dataset_name="NSL-KDD"):
    """
    Preprocess input dataframe to match model requirements based on dataset.
    """

    if dataset_name == "NSL-KDD":
        # Ensure all categorical columns exist in input
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

    elif dataset_name == "UNSW-NB15":
        # Label Encoding using saved encoders
        if encoders:
            for col, le in encoders.items():
                if col in df.columns:
                    # Handle unknown values
                    df[col] = df[col].fillna("unknown").astype(str)
                    df[col] = df[col].apply(
                        lambda x: x if x in le.classes_ else "unknown"
                    )
                    # Make sure unknown is in classes (it should be from generation)
                    df[col] = le.transform(df[col])

        # Ensure correct columns match feature_cols
        # Add missing columns with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Select columns in order
        X = df[feature_cols].values

    elif dataset_name == "CICIDS2017":
        # Just select columns and fill missing
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Handle Infinity and NaN which are common in CICIDS2017
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        X = df[feature_cols].values

    else:
        return None

    # Scale
    X_scaled = scaler.transform(X)

    return X_scaled
