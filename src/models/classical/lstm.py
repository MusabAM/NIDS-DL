"""
LSTM (Long Short-Term Memory) models for NIDS classification.
Implementations in both TensorFlow/Keras and PyTorch.
"""

from typing import List, Optional
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ==============================================================================
# PyTorch LSTM Model
# ==============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for NIDS (PyTorch).
    
    Architecture:
        - Bidirectional LSTM layers
        - Dropout for regularization
        - Fully connected layers for classification
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lstm_units: List[int] = [128, 64],
        dense_units: List[int] = [128, 64],
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
    ):
        """
        Initialize LSTM Classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            lstm_units: List of hidden units for each LSTM layer
            dense_units: List of units for dense layers
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout probability
            recurrent_dropout: Recurrent dropout (applied between LSTM layers)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        in_features = input_dim
        
        for i, units in enumerate(lstm_units):
            lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=units,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=recurrent_dropout if i < len(lstm_units) - 1 else 0,
            )
            self.lstm_layers.append(lstm)
            in_features = units * self.num_directions
        
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        dense_layers = []
        in_features = lstm_units[-1] * self.num_directions
        
        for units in dense_units:
            dense_layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_features = units
        
        self.dense_block = nn.Sequential(*dense_layers)
        
        # Output layer
        self.output_layer = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Reshape for LSTM: (batch, features) -> (batch, seq_len=1, features)
        # Or treat features as sequence: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(1)  # (batch, 1, features) - single timestep
        
        # LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Take last output
        x = x[:, -1, :]  # (batch, hidden_size * num_directions)
        x = self.lstm_dropout(x)
        
        # Dense block
        x = self.dense_block(x)
        
        # Output
        x = self.output_layer(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def create_lstm_torch(
    input_dim: int,
    num_classes: int,
    **kwargs
) -> LSTMClassifier:
    """Factory function to create PyTorch LSTM model."""
    return LSTMClassifier(input_dim, num_classes, **kwargs)


# ==============================================================================
# TensorFlow/Keras LSTM Model
# ==============================================================================

def create_lstm_tf(
    input_dim: int,
    num_classes: int,
    lstm_units: List[int] = [128, 64],
    dense_units: List[int] = [128, 64],
    bidirectional: bool = True,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> "keras.Model":
    """
    Create an LSTM model using TensorFlow/Keras.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        lstm_units: List of units for each LSTM layer
        dense_units: List of units for dense layers
        bidirectional: Whether to use bidirectional LSTM
        dropout_rate: Dropout rate
        recurrent_dropout: Recurrent dropout rate
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for create_lstm_tf")
    
    # Input layer
    inputs = keras.Input(shape=(input_dim,), name="input")
    
    # Reshape for LSTM: (batch, features) -> (batch, timesteps=1, features)
    x = layers.Reshape((1, input_dim))(inputs)
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1  # Return sequences for all but last LSTM
        
        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
        )
        
        if bidirectional:
            x = layers.Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)
    
    # Dense layers
    for units in dense_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', keras.metrics.AUC(name='auc')]
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_Classifier")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    
    return model
