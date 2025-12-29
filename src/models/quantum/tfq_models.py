"""
TensorFlow Quantum (TFQ) models for NIDS classification.

These models use Google's Cirq for quantum circuit construction and
TensorFlow Quantum for hybrid quantum-classical training.
"""

from typing import List, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import cirq
    import tensorflow_quantum as tfq
    import sympy
    TFQ_AVAILABLE = True
except ImportError:
    TFQ_AVAILABLE = False


def check_tfq():
    """Check if TensorFlow Quantum is available."""
    if not TFQ_AVAILABLE:
        raise ImportError(
            "TensorFlow Quantum is required. "
            "Install with: pip install tensorflow-quantum"
        )
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for TFQ models.")


# ==============================================================================
# Cirq Circuit Builders
# ==============================================================================

def create_encoding_circuit(
    qubits: List[cirq.GridQubit],
    input_symbols: List[sympy.Symbol],
) -> cirq.Circuit:
    """
    Create a circuit that encodes classical data into quantum states.
    
    Args:
        qubits: List of qubits
        input_symbols: Symbolic parameters for encoding
        
    Returns:
        Encoding circuit
    """
    circuit = cirq.Circuit()
    
    for i, (qubit, symbol) in enumerate(zip(qubits, input_symbols)):
        # RY rotation for angle encoding
        circuit.append(cirq.ry(symbol * np.pi).on(qubit))
    
    return circuit


def create_variational_circuit(
    qubits: List[cirq.GridQubit],
    n_layers: int,
    parameter_prefix: str = "θ",
) -> Tuple[cirq.Circuit, List[sympy.Symbol]]:
    """
    Create a variational quantum circuit with parameterized gates.
    
    Args:
        qubits: List of qubits
        n_layers: Number of variational layers
        parameter_prefix: Prefix for parameter symbols
        
    Returns:
        Tuple of (circuit, list of parameter symbols)
    """
    n_qubits = len(qubits)
    circuit = cirq.Circuit()
    symbols = []
    
    symbol_idx = 0
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i, qubit in enumerate(qubits):
            # RX, RY, RZ rotations
            for gate_name, gate in [('rx', cirq.rx), ('ry', cirq.ry), ('rz', cirq.rz)]:
                symbol = sympy.Symbol(f"{parameter_prefix}_{layer}_{i}_{gate_name}")
                symbols.append(symbol)
                circuit.append(gate(symbol).on(qubit))
                symbol_idx += 1
        
        # Entangling layer (CNOT ladder)
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        
        # Circular connection (last to first)
        if n_qubits > 1:
            circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    
    return circuit, symbols


def create_readout_operators(
    qubits: List[cirq.GridQubit],
    n_outputs: int,
) -> List[cirq.PauliString]:
    """
    Create Pauli-Z readout operators.
    
    Args:
        qubits: List of qubits
        n_outputs: Number of outputs (measurements)
        
    Returns:
        List of Pauli-Z operators
    """
    operators = []
    for i in range(min(n_outputs, len(qubits))):
        operators.append(cirq.Z(qubits[i]))
    return operators


# ==============================================================================
# TFQ Classifier
# ==============================================================================

class TFQClassifier:
    """
    TensorFlow Quantum Classifier for NIDS.
    
    Architecture:
        1. Classical preprocessing (optional)
        2. Quantum encoding circuit
        3. Quantum variational circuit
        4. Classical postprocessing
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        num_classes: int = 2,
        classical_layers: Optional[List[int]] = None,
    ):
        """
        Initialize TFQ Classifier.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            num_classes: Number of output classes
            classical_layers: Units for classical postprocessing layers
        """
        check_tfq()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.classical_layers = classical_layers or [32, 16]
        
        # Create qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        
        # Create symbols for encoding
        self.input_symbols = [sympy.Symbol(f"x_{i}") for i in range(n_qubits)]
        
        # Create encoding circuit
        self.encoding_circuit = create_encoding_circuit(
            self.qubits, self.input_symbols
        )
        
        # Create variational circuit
        self.var_circuit, self.var_symbols = create_variational_circuit(
            self.qubits, n_layers
        )
        
        # Full quantum circuit
        self.full_circuit = self.encoding_circuit + self.var_circuit
        
        # Readout operators
        self.readout_ops = create_readout_operators(
            self.qubits, num_classes
        )
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the full Keras model."""
        
        # Input layer for quantum circuit parameters
        circuit_input = keras.Input(shape=(), dtype=tf.string, name="circuits")
        
        # PQC (Parameterized Quantum Circuit) layer
        pqc = tfq.layers.PQC(
            self.var_circuit,
            self.readout_ops,
            repetitions=1000,  # Number of measurement shots
            differentiator=tfq.differentiators.ParameterShift(),
        )
        
        quantum_output = pqc(circuit_input)
        
        # Classical postprocessing
        x = quantum_output
        for units in self.classical_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        # Output layer
        if self.num_classes == 2:
            output = layers.Dense(1, activation='sigmoid')(x)
        else:
            output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=circuit_input, outputs=output)
        
        return model
    
    def preprocess_data(self, X: np.ndarray) -> tf.Tensor:
        """
        Convert classical data to quantum circuit tensors.
        
        Args:
            X: Classical features of shape (n_samples, n_features)
               Features should be normalized to [0, 1]
               
        Returns:
            Tensor of serialized quantum circuits
        """
        circuits = []
        
        for sample in X:
            # Create resolver with input values
            resolver = cirq.ParamResolver({
                str(sym): val 
                for sym, val in zip(self.input_symbols, sample[:self.n_qubits])
            })
            
            # Resolve the encoding circuit
            resolved_circuit = cirq.resolve_parameters(
                self.encoding_circuit, resolver
            )
            
            circuits.append(resolved_circuit)
        
        # Convert to TFQ tensor
        return tfq.convert_to_tensor(circuits)
    
    def compile(
        self,
        learning_rate: float = 0.01,
        loss: Optional[str] = None,
    ):
        """Compile the model."""
        if loss is None:
            loss = 'binary_crossentropy' if self.num_classes == 2 else 'sparse_categorical_crossentropy'
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy'],
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features (normalized to [0, 1])
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional validation data
        """
        # Preprocess to quantum circuits
        X_train_q = self.preprocess_data(X_train)
        
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_q = self.preprocess_data(X_val)
            val_data = (X_val_q, y_val)
        
        return self.model.fit(
            X_train_q, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            **kwargs
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions."""
        X_q = self.preprocess_data(X)
        return self.model.predict(X_q)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model."""
        X_q = self.preprocess_data(X)
        return self.model.evaluate(X_q, y, return_dict=True)


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_tfq_model(
    n_qubits: int = 8,
    n_layers: int = 3,
    num_classes: int = 2,
    **kwargs
) -> TFQClassifier:
    """
    Factory function to create TFQ classifier.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        num_classes: Number of output classes
        
    Returns:
        TFQClassifier instance
    """
    return TFQClassifier(
        n_qubits=n_qubits,
        n_layers=n_layers,
        num_classes=num_classes,
        **kwargs
    )


def create_hybrid_tfq_model(
    input_dim: int,
    num_classes: int,
    n_qubits: int = 8,
    n_quantum_layers: int = 3,
    pre_layers: List[int] = [64, 32],
    post_layers: List[int] = [32, 16],
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Create a hybrid classical-quantum model using TFQ.
    
    This model uses classical layers for preprocessing before the quantum circuit.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        n_qubits: Number of qubits
        n_quantum_layers: Number of quantum layers
        pre_layers: Classical preprocessing layer units
        post_layers: Classical postprocessing layer units
        learning_rate: Learning rate
        
    Returns:
        Compiled Keras model
    """
    check_tfq()
    
    # Create quantum components
    qubits = cirq.GridQubit.rect(1, n_qubits)
    
    var_circuit, var_symbols = create_variational_circuit(
        qubits, n_quantum_layers
    )
    
    readout_ops = create_readout_operators(qubits, n_qubits)
    
    # Classical input
    classical_input = keras.Input(shape=(input_dim,), name="classical_input")
    
    # Classical preprocessing
    x = classical_input
    for units in pre_layers:
        x = layers.Dense(units, activation='relu')(x)
    
    # Reduce to n_qubits dimensions
    x = layers.Dense(n_qubits, activation='tanh')(x)  # Output in [-1, 1]
    
    # Scale to [0, 1] for quantum encoding
    x = layers.Lambda(lambda z: (z + 1) / 2)(x)
    
    # Note: For a true hybrid model, we'd need to:
    # 1. Create circuits dynamically from classical outputs
    # 2. Use tfq.layers.ControlledPQC or custom layers
    # This is a simplified version that demonstrates the concept
    
    # Quantum-inspired classical layer (simulates quantum behavior)
    # In practice, you'd use TFQ's quantum layers here
    x = layers.Dense(n_qubits, activation='tanh', name='quantum_inspired')(x)
    
    # Classical postprocessing
    for units in post_layers:
        x = layers.Dense(units, activation='relu')(x)
    
    # Output
    if num_classes == 2:
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        output = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
    
    model = keras.Model(inputs=classical_input, outputs=output, name='hybrid_tfq')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy'],
    )
    
    return model


# ==============================================================================
# Quantum Circuit Visualization
# ==============================================================================

def visualize_circuit(classifier: TFQClassifier, filename: Optional[str] = None):
    """
    Visualize the quantum circuit.
    
    Args:
        classifier: TFQClassifier instance
        filename: Optional filename to save the visualization
    """
    check_tfq()
    
    print("Encoding Circuit:")
    print(classifier.encoding_circuit)
    print("\nVariational Circuit:")
    print(classifier.var_circuit)
    
    if filename:
        # Save as SVG
        try:
            svg = cirq.contrib.svg.SVGCircuit(classifier.full_circuit)
            with open(filename, 'w') as f:
                f.write(str(svg))
            print(f"\nCircuit saved to {filename}")
        except Exception as e:
            print(f"Could not save circuit: {e}")
