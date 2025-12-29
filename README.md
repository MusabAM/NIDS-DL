# NIDS-DL: Network Intrusion Detection using Deep Learning & Quantum ML

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.34+-green.svg)](https://pennylane.ai/)

A comprehensive research framework for exploring classical deep learning and quantum machine learning approaches to Network Intrusion Detection Systems (NIDS).

## 🚀 Features

- **Multiple Deep Learning Architectures**: CNN, LSTM, Transformer, Autoencoder
- **Quantum Machine Learning**: PennyLane and TensorFlow Quantum implementations
- **Multiple Datasets**: NSL-KDD, CICIDS2017, CICIDS2018, UNSW-NB15
- **GPU Acceleration**: Full CUDA support for NVIDIA GPUs
- **Comprehensive Evaluation**: Metrics, visualizations, and comparison tools
- **Modular Design**: Easy to extend with new models and datasets

## 📋 System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11, Linux, macOS |
| Python | 3.11 or 3.12 (recommended) |
| GPU | NVIDIA with CUDA 12.x (optional but recommended) |
| RAM | 16GB+ (32GB recommended) |

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nids-dl.git
cd nids-dl
```

### 2. Create Virtual Environment

```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install quantum dependencies (optional)
pip install -r requirements-quantum.txt
```

### 4. Verify Installation

```bash
python scripts/verify_setup.py
```

## 📊 Datasets

### Download NSL-KDD (Quick Start)

```bash
python scripts/download_datasets.py --dataset nsl_kdd
```

### Other Datasets

| Dataset | Size | Download |
|---------|------|----------|
| NSL-KDD | ~10MB | Automatic |
| CICIDS2017 | ~6GB | [Manual Download](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CICIDS2018 | ~16GB | [Manual Download](https://www.unb.ca/cic/datasets/ids-2018.html) |
| UNSW-NB15 | ~1GB | [Manual Download](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

## 🏗️ Project Structure

```
nids-dl/
├── .vscode/                    # VS Code configuration
├── configs/                    # YAML configuration files
│   ├── datasets.yaml           # Dataset configurations
│   ├── models.yaml             # Model architectures
│   └── training.yaml           # Training parameters
├── data/                       # Dataset storage
│   ├── raw/                    # Original datasets
│   └── processed/              # Preprocessed data
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration/    # Dataset analysis
│   ├── 02_classical_dl/        # DL experiments
│   ├── 03_quantum_dl/          # Quantum ML experiments
│   └── 04_experiments/         # Custom experiments
├── src/                        # Source code
│   ├── data/                   # Data loading & preprocessing
│   ├── models/                 # Model implementations
│   │   ├── classical/          # CNN, LSTM, Transformer, Autoencoder
│   │   └── quantum/            # PennyLane, TFQ models
│   ├── training/               # Training utilities
│   ├── evaluation/             # Metrics & visualization
│   └── utils/                  # Configuration & logging
├── scripts/                    # Utility scripts
├── results/                    # Experiment outputs
│   ├── models/                 # Saved model weights
│   ├── logs/                   # Training logs
│   └── figures/                # Generated plots
├── requirements.txt            # Core dependencies
└── requirements-quantum.txt    # Quantum dependencies
```

## 🚀 Quick Start

### 1. Train a CNN Classifier

```python
import torch
from src.data import get_dataset, get_dataloaders
from src.models import CNNClassifier
from src.training import TorchTrainer

# Load and preprocess data
data = get_dataset("nsl_kdd", classification="binary")

# Create data loaders
train_loader, val_loader, test_loader = get_dataloaders(
    data["X_train"], data["y_train"],
    data["X_val"], data["y_val"],
    data["X_test"], data["y_test"],
    batch_size=256
)

# Create model
model = CNNClassifier(
    input_dim=data["info"].num_features,
    num_classes=data["info"].num_classes
)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = TorchTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    criterion=torch.nn.CrossEntropyLoss(),
    device=device
)

history = trainer.train(train_loader, val_loader, epochs=50)
results = trainer.evaluate(test_loader)
```

### 2. Train a Quantum Classifier

```python
from src.models.quantum import HybridQuantumClassifier

# Create hybrid quantum-classical model
model = HybridQuantumClassifier(
    input_dim=41,  # NSL-KDD features
    num_classes=2,
    n_qubits=8,
    n_quantum_layers=4,
)

# Training follows the same pattern as classical models
```

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_nsl_kdd_exploration.ipynb` | NSL-KDD data analysis and visualization |
| `02_cicids_exploration.ipynb` | CICIDS2017/2018 dataset exploration |
| `03_classical_baseline.ipynb` | Traditional ML baselines (RF, SVM, XGBoost) |
| `04_deep_learning_models.ipynb` | CNN, LSTM, Transformer training |
| `05_quantum_pennylane.ipynb` | PennyLane quantum experiments |
| `06_model_comparison.ipynb` | Performance comparison across all models |

## 🔬 Model Architectures

### Classical Models

| Model | Description | Best For |
|-------|-------------|----------|
| CNN | 1D Convolutions for feature extraction | Spatial patterns |
| LSTM | Bidirectional LSTM for sequences | Temporal patterns |
| Transformer | Self-attention mechanism | Complex dependencies |
| Autoencoder | Reconstruction-based anomaly detection | Unsupervised detection |

### Quantum Models

| Model | Framework | Description |
|-------|-----------|-------------|
| VQC | PennyLane | Variational Quantum Classifier |
| Hybrid | PennyLane | Classical pre/post-processing + quantum |
| TFQ | TensorFlow Quantum | Cirq-based quantum circuits |

## 📈 Results

Results are saved in the `results/` directory:

- **Models**: `results/models/*.pt` (PyTorch), `results/models/*.h5` (Keras)
- **Logs**: `results/logs/*.log`
- **Figures**: `results/figures/*.png`
- **Metrics**: `results/logs/*_metrics.json`

## 🛠️ Configuration

All configurations are in YAML format under `configs/`:

```yaml
# configs/training.yaml
hyperparameters:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001
  optimizer: adam

early_stopping:
  patience: 15
  monitor: val_accuracy
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

### Datasets
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CICIDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### Libraries
- [TensorFlow](https://tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [PennyLane](https://pennylane.ai/)
- [TensorFlow Quantum](https://www.tensorflow.org/quantum)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✉️ Contact

For questions or collaboration, please open an issue or contact the maintainers.

---

**Happy researching! 🎯**
