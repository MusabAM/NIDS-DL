"""
NIDS-DL: Network Intrusion Detection using Deep Learning and Quantum ML
========================================================================

A comprehensive research framework for exploring classical and quantum
machine learning approaches to network intrusion detection.

Modules:
    - data: Dataset loading, preprocessing, and augmentation
    - models: Classical (CNN, LSTM, Transformer) and Quantum (PennyLane, TFQ) models
    - training: Training loops, callbacks, and optimization
    - evaluation: Metrics, visualization, and model comparison
    - utils: Configuration, logging, and utility functions
"""

__version__ = "0.1.0"
__author__ = "NIDS-DL Research"

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
