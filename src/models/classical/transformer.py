"""
Transformer-based model for NIDS classification.
Implementation in PyTorch with attention mechanism.
"""

from typing import List, Optional
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for NIDS.
    
    Architecture:
        - Input embedding layer
        - Positional encoding
        - Multiple transformer encoder blocks
        - Global average pooling
        - Classification head
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        num_blocks: int = 4,
        dense_units: List[int] = [128],
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer Classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_blocks: Number of transformer blocks
            dense_units: Units for classification head
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(1, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=input_dim, dropout=dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Classification head
        head_layers = []
        in_features = embed_dim
        
        for units in dense_units:
            head_layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = units
        
        self.classification_head = nn.Sequential(*head_layers)
        self.output_layer = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Treat each feature as a sequence element: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        
        # Embed each feature
        x = self.input_embedding(x)  # (batch, features, embed_dim)
        
        # Transpose for transformer: (batch, seq, embed) -> (seq, batch, embed)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Transpose back: (seq, batch, embed) -> (batch, seq, embed)
        x = x.transpose(0, 1)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, embed_dim)
        
        # Classification head
        x = self.classification_head(x)
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
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from all transformer blocks.
        Useful for interpretability analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weight tensors
        """
        attention_weights = []
        
        x = x.unsqueeze(-1)
        x = self.input_embedding(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            _, weights = block.attention(x, x, x, need_weights=True)
            attention_weights.append(weights)
            x = block(x)
        
        return attention_weights


def create_transformer(
    input_dim: int,
    num_classes: int,
    **kwargs
) -> TransformerClassifier:
    """Factory function to create Transformer model."""
    return TransformerClassifier(input_dim, num_classes, **kwargs)
