"""
PennyLane-based quantum machine learning models for NIDS.

These models implement variational quantum classifiers (VQC) that can be
integrated with classical neural networks for hybrid quantum-classical learning.
"""

from typing import List, Optional, Callable, Tuple
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_pennylane():
    """Check if PennyLane is available."""
    if not PENNYLANE_AVAILABLE:
        raise ImportError(
            "PennyLane is required for quantum models. "
            "Install with: pip install pennylane"
        )


# ==============================================================================
# Quantum Circuits
# ==============================================================================

def angle_encoding(features: np.ndarray, wires: List[int]):
    """
    Encode classical features as rotation angles.
    
    Args:
        features: Input features (should be normalized to [0, 2π])
        wires: Qubit indices to use
    """
    for i, wire in enumerate(wires):
        if i < len(features):
            qml.RY(features[i], wires=wire)


def amplitude_encoding(features: np.ndarray, wires: List[int]):
    """
    Encode classical features as quantum amplitudes.
    
    Args:
        features: Input features (will be normalized)
        wires: Qubit indices to use
    """
    # Pad to 2^n length
    n_qubits = len(wires)
    n_amplitudes = 2 ** n_qubits
    
    if len(features) < n_amplitudes:
        features = np.pad(features, (0, n_amplitudes - len(features)))
    
    # Normalize
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    qml.AmplitudeEmbedding(features[:n_amplitudes], wires=wires, normalize=True)


def basic_entangling_layer(params: np.ndarray, wires: List[int]):
    """
    Basic entangling layer with rotation gates and CNOTs.
    
    Args:
        params: Rotation parameters of shape (n_qubits, 3)
        wires: Qubit indices
    """
    n_qubits = len(wires)
    
    # Rotation gates
    for i, wire in enumerate(wires):
        qml.RX(params[i, 0], wires=wire)
        qml.RY(params[i, 1], wires=wire)
        qml.RZ(params[i, 2], wires=wire)
    
    # CNOT ladder
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    
    # Circular entanglement (last to first)
    if n_qubits > 1:
        qml.CNOT(wires=[wires[-1], wires[0]])


def strongly_entangling_layer(params: np.ndarray, wires: List[int]):
    """
    Strongly entangling layer using PennyLane's built-in template.
    
    Args:
        params: Rotation parameters of shape (n_qubits, 3)
        wires: Qubit indices
    """
    qml.StronglyEntanglingLayers(params.reshape(1, len(wires), 3), wires=wires)


# ==============================================================================
# Quantum Classifier (PennyLane + PyTorch)
# ==============================================================================

class QuantumClassifier(nn.Module):
    """
    Variational Quantum Classifier using PennyLane.
    
    This is a pure quantum classifier where:
    - Classical features are encoded into quantum states
    - Variational layers perform quantum transformations
    - Measurements are used for classification
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 4,
        n_classes: int = 2,
        encoding: str = "angle",
        entangling: str = "strongly_entangling",
        device: str = "default.qubit",
    ):
        """
        Initialize Quantum Classifier.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            n_classes: Number of output classes
            encoding: Feature encoding method ('angle' or 'amplitude')
            entangling: Entangling layer type ('basic' or 'strongly_entangling')
            device: PennyLane device to use
        """
        check_pennylane()
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # Create quantum device
        self.dev = qml.device(device, wires=n_qubits)
        
        # Choose encoding method
        if encoding == "angle":
            self.encode = angle_encoding
        elif encoding == "amplitude":
            self.encode = amplitude_encoding
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        
        # Choose entangling layer
        if entangling == "basic":
            self.entangle = basic_entangling_layer
        elif entangling == "strongly_entangling":
            self.entangle = lambda p, w: qml.StronglyEntanglingLayers(p, wires=w)
        else:
            raise ValueError(f"Unknown entangling: {entangling}")
        
        # Variational parameters
        # Shape: (n_layers, n_qubits, 3) for rotation angles
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(weight_shape) * 0.1
        )
        
        # Create quantum node
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Encode classical data
            wires = list(range(self.n_qubits))
            
            # Angle encoding
            for i, wire in enumerate(wires):
                if i < len(inputs):
                    qml.RY(inputs[i] * np.pi, wires=wire)
            
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=wires)
            
            # Measurements (expectation values of Pauli-Z)
            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_classes, n_qubits))]
        
        self.circuit = circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum circuit.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
               Features should be normalized to [0, 1]
               
        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Get expectation values from quantum circuit
            result = self.circuit(x[i, :self.n_qubits], self.weights)
            
            # Stack results
            if isinstance(result, list):
                result = torch.stack(result)
            outputs.append(result)
        
        return torch.stack(outputs)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


class HybridQuantumClassifier(nn.Module):
    """
    Hybrid Classical-Quantum Classifier.
    
    Architecture:
        1. Classical preprocessing layers (dimensionality reduction)
        2. Quantum variational circuit
        3. Classical postprocessing layers
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_qubits: int = 8,
        n_quantum_layers: int = 4,
        pre_layers: List[int] = [64, 32],
        post_layers: List[int] = [32, 16],
        dropout: float = 0.2,
        device: str = "default.qubit",
    ):
        """
        Initialize Hybrid Quantum Classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            n_qubits: Number of qubits in quantum circuit
            n_quantum_layers: Number of variational layers
            pre_layers: Units for classical preprocessing layers
            post_layers: Units for classical postprocessing layers
            dropout: Dropout rate for classical layers
            device: PennyLane quantum device
        """
        check_pennylane()
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Classical preprocessing
        pre = []
        in_features = input_dim
        for units in pre_layers:
            pre.extend([
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = units
        
        # Final layer to match qubits
        pre.append(nn.Linear(in_features, n_qubits))
        pre.append(nn.Tanh())  # Normalize to [-1, 1]
        
        self.preprocessing = nn.Sequential(*pre)
        
        # Quantum circuit
        dev = qml.device(device, wires=n_qubits)
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            wires = list(range(n_qubits))
            
            # Encode preprocessed features
            for i, wire in enumerate(wires):
                qml.RY(inputs[i] * np.pi, wires=wire)
            
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=wires)
            
            # Return all qubit expectation values
            return [qml.expval(qml.PauliZ(i)) for i in wires]
        
        self.quantum_circuit = quantum_circuit
        
        # Quantum parameters
        self.quantum_weights = nn.Parameter(
            torch.randn(n_quantum_layers, n_qubits, 3) * 0.1
        )
        
        # Classical postprocessing
        post = []
        in_features = n_qubits
        for units in post_layers:
            post.extend([
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = units
        
        post.append(nn.Linear(in_features, num_classes))
        self.postprocessing = nn.Sequential(*post)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Classical preprocessing
        x = self.preprocessing(x)  # (batch, n_qubits)
        
        # Quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            result = self.quantum_circuit(x[i], self.quantum_weights)
            if isinstance(result, list):
                result = torch.stack(result)
            quantum_outputs.append(result)
        
        x = torch.stack(quantum_outputs)  # (batch, n_qubits)
        
        # Classical postprocessing
        x = self.postprocessing(x)
        
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


# ==============================================================================
# Utility Functions
# ==============================================================================

def create_quantum_layer(
    n_qubits: int,
    n_layers: int = 2,
    device: str = "default.qubit",
) -> Callable:
    """
    Create a quantum layer that can be used in a classical neural network.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        device: PennyLane device
        
    Returns:
        Quantum layer function
    """
    check_pennylane()
    
    dev = qml.device(device, wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_layer(inputs, weights):
        wires = list(range(n_qubits))
        
        # Encode inputs
        for i, wire in enumerate(wires):
            qml.RY(inputs[i], wires=wire)
        
        # Variational layers
        qml.StronglyEntanglingLayers(weights, wires=wires)
        
        return [qml.expval(qml.PauliZ(i)) for i in wires]
    
    return quantum_layer


def quantum_embedding(
    features: np.ndarray,
    n_qubits: int,
    method: str = "angle",
) -> np.ndarray:
    """
    Create quantum embeddings for classical features.
    
    Args:
        features: Classical features
        n_qubits: Number of qubits
        method: Encoding method
        
    Returns:
        Quantum-embedded features
    """
    check_pennylane()
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def embed(x):
        for i in range(n_qubits):
            if i < len(x):
                qml.RY(x[i] * np.pi, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return np.array(embed(features))
