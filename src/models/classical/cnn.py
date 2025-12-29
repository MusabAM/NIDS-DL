"""
CNN (Convolutional Neural Network) models for NIDS classification.
Implementations in both TensorFlow/Keras and PyTorch.
"""

from typing import List, Optional, Tuple
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
# PyTorch CNN Model
# ==============================================================================

class CNNClassifier(nn.Module):
    """
    1D Convolutional Neural Network for NIDS classification (PyTorch).
    
    Architecture:
        - Multiple 1D Conv layers with BatchNorm and MaxPool
        - Dropout for regularization
        - Fully connected layers for classification
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3],
        dense_units: List[int] = [256, 128],
        dropout_rate: float = 0.3,
    ):
        """
        Initialize CNN Classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            conv_channels: List of channels for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dense_units: List of units for dense layers
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Convolutional layers
        conv_layers = []
        in_channels = 1  # Single channel for 1D features
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate),
            ])
            in_channels = out_channels
        
        self.conv_block = nn.Sequential(*conv_layers)
        
        # Calculate flattened size after convolutions
        self._conv_output_size = self._get_conv_output_size(input_dim)
        
        # Dense layers
        dense_layers = []
        in_features = self._conv_output_size
        
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
    
    def _get_conv_output_size(self, input_dim: int) -> int:
        """Calculate the output size after convolutional layers."""
        with torch.no_grad():
            x = torch.zeros(1, 1, input_dim)
            x = self.conv_block(x)
            return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Reshape for 1D convolution: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional block
        x = self.conv_block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
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


def create_cnn_torch(
    input_dim: int,
    num_classes: int,
    **kwargs
) -> CNNClassifier:
    """Factory function to create PyTorch CNN model."""
    return CNNClassifier(input_dim, num_classes, **kwargs)


# ==============================================================================
# TensorFlow/Keras CNN Model
# ==============================================================================

def create_cnn_tf(
    input_dim: int,
    num_classes: int,
    conv_filters: List[int] = [64, 128, 256],
    kernel_sizes: List[int] = [3, 3, 3],
    dense_units: List[int] = [256, 128],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> "keras.Model":
    """
    Create a 1D CNN model using TensorFlow/Keras.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        conv_filters: List of filters for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        dense_units: List of units for dense layers
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for create_cnn_tf")
    
    # Input layer (reshape for 1D conv)
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Convolutional blocks
    for filters, kernel_size in zip(conv_filters, kernel_sizes):
        x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
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
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_Classifier")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    
    return model


# ==============================================================================
# Model Summary & Utilities
# ==============================================================================

def get_model_summary(model, input_dim: int, framework: str = "pytorch"):
    """Print model summary."""
    if framework == "pytorch":
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    else:
        model.summary()
