"""
Autoencoder model for anomaly-based NIDS.
Uses reconstruction error for anomaly detection.
"""

from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ==============================================================================
# PyTorch Autoencoder
# ==============================================================================

class Autoencoder(nn.Module):
    """
    Autoencoder for anomaly detection in NIDS.
    
    Architecture:
        - Encoder: Compresses input to latent representation
        - Decoder: Reconstructs input from latent representation
        
    Anomaly detection is based on reconstruction error:
    - Train on normal traffic only
    - High reconstruction error indicates anomaly
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_units: List[int] = [64, 32, 16],
        latent_dim: int = 8,
        decoder_units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        activation: str = "relu",
    ):
        """
        Initialize Autoencoder.
        
        Args:
            input_dim: Number of input features
            encoder_units: List of units for encoder layers
            latent_dim: Dimension of latent space
            decoder_units: List of units for decoder (default: reverse of encoder)
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
        """
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
            encoder_layers.extend([
                nn.Linear(in_features, units),
                nn.BatchNorm1d(units),
                self.activation,
                nn.Dropout(dropout_rate),
            ])
            in_features = units
        
        encoder_layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        in_features = latent_dim
        
        for units in decoder_units:
            decoder_layers.extend([
                nn.Linear(in_features, units),
                nn.BatchNorm1d(units),
                self.activation,
                nn.Dropout(dropout_rate),
            ])
            in_features = units
        
        decoder_layers.append(nn.Linear(in_features, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1] for normalized features
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Threshold for anomaly detection (set during training)
        self.threshold = None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error (MSE) for each sample.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstruction error for each sample
        """
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            mse = F.mse_loss(x_recon, x, reduction='none')
            return mse.mean(dim=1)  # Mean error per sample
    
    def set_threshold(
        self,
        X_normal: torch.Tensor,
        percentile: float = 95,
    ) -> float:
        """
        Set anomaly detection threshold based on normal data.
        
        Args:
            X_normal: Normal training data
            percentile: Percentile of reconstruction errors to use as threshold
            
        Returns:
            Threshold value
        """
        errors = self.reconstruction_error(X_normal)
        self.threshold = np.percentile(errors.cpu().numpy(), percentile)
        return self.threshold
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict whether samples are anomalies.
        
        Args:
            x: Input tensor
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        errors = self.reconstruction_error(x)
        return (errors > self.threshold).long()
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly scores (reconstruction errors).
        
        Args:
            x: Input tensor
            
        Returns:
            Anomaly scores for each sample
        """
        return self.reconstruction_error(x)


class SupervisedAutoencoder(nn.Module):
    """
    Supervised Autoencoder for simultaneous reconstruction and classification.
    
    Architecture:
        - Encoder: Input -> Latent
        - Decoder: Latent -> Reconstruction
        - Classifier: Latent -> Class Prediction
        
    Forward returns: (reconstruction, logits, latent)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        encoder_units: List[int] = [64, 32, 16],
        latent_dim: int = 8,
        decoder_units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        activation: str = "relu",
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
            encoder_layers.extend([
                nn.Linear(in_features, units),
                nn.BatchNorm1d(units),
                self.activation,
                nn.Dropout(dropout_rate),
            ])
            in_features = units
        
        encoder_layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        in_features = latent_dim
        
        for units in decoder_units:
            decoder_layers.extend([
                nn.Linear(in_features, units),
                nn.BatchNorm1d(units),
                self.activation,
                nn.Dropout(dropout_rate),
            ])
            in_features = units
        
        decoder_layers.append(nn.Linear(in_features, input_dim))
        decoder_layers.append(nn.Sigmoid()) 
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Build Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (reconstruction, logits, latent_vector)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        logits = self.classifier(z)
        return x_recon, logits, z
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        with torch.no_grad():
            _, logits, _ = self.forward(x)
            return torch.argmax(logits, dim=1)


# ==============================================================================
# TensorFlow/Keras Autoencoder
# ==============================================================================

def create_autoencoder_tf(
    input_dim: int,
    encoder_units: List[int] = [64, 32, 16],
    latent_dim: int = 8,
    decoder_units: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Tuple["keras.Model", "keras.Model", "keras.Model"]:
    """
    Create Autoencoder using TensorFlow/Keras.
    
    Args:
        input_dim: Number of input features
        encoder_units: List of units for encoder layers
        latent_dim: Dimension of latent space
        decoder_units: List of units for decoder
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (autoencoder, encoder, decoder) models
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for create_autoencoder_tf")
    
    if decoder_units is None:
        decoder_units = list(reversed(encoder_units))
    
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,), name="encoder_input")
    x = encoder_input
    
    for units in encoder_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    latent = layers.Dense(latent_dim, name="latent")(x)
    encoder = keras.Model(encoder_input, latent, name="encoder")
    
    # Decoder
    decoder_input = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = decoder_input
    
    for units in decoder_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    decoder_output = layers.Dense(input_dim, activation='sigmoid', name="decoder_output")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")
    
    # Full autoencoder
    autoencoder_input = keras.Input(shape=(input_dim,), name="autoencoder_input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
    )
    
    return autoencoder, encoder, decoder


# ==============================================================================
# Variational Autoencoder (VAE)
# ==============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder for anomaly detection.
    
    Uses probabilistic latent space with reparameterization trick.
    Anomaly detection uses ELBO (Evidence Lower Bound) as score.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_units: List[int] = [64, 32],
        latent_dim: int = 8,
        decoder_units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if decoder_units is None:
            decoder_units = list(reversed(encoder_units))
        
        # Encoder
        encoder_layers = []
        in_features = input_dim
        
        for units in encoder_units:
            encoder_layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_features = units
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)
        
        # Decoder
        decoder_layers = [nn.Linear(latent_dim, decoder_units[0]), nn.ReLU()]
        in_features = decoder_units[0]
        
        for units in decoder_units[1:]:
            decoder_layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_features = units
        
        decoder_layers.append(nn.Linear(in_features, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input and return mean and log variance."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backpropagation through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (reconstruction, mean, log_variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss = Reconstruction loss + KL divergence.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            beta: Weight for KL divergence (beta-VAE)
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_divergence)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_div
        
        return total_loss, recon_loss, kl_div
