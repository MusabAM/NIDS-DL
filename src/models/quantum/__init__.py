"""Quantum machine learning models."""

from .pennylane_models import (
    HybridQuantumClassifier,
    QuantumClassifier,
    create_quantum_layer,
)

# TFQ imports are optional
try:
    from .tfq_models import TFQClassifier, create_tfq_model

    _TFQ_AVAILABLE = True
except ImportError:
    _TFQ_AVAILABLE = False
    TFQClassifier = None
    create_tfq_model = None

__all__ = [
    "QuantumClassifier",
    "HybridQuantumClassifier",
    "create_quantum_layer",
]

# Add TFQ exports only if available
if _TFQ_AVAILABLE:
    __all__.extend(
        [
            "TFQClassifier",
            "create_tfq_model",
        ]
    )
