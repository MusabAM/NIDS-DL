# Quantum Deep Learning Notebooks

This directory contains Jupyter notebooks for implementing **Variational Quantum Classifiers (VQC)** for Network Intrusion Detection using PennyLane.

## Overview

The notebooks implement a **Hybrid Quantum-Classical** approach:
- **Classical Preprocessing**: PCA dimensionality reduction (122 → 8 features)
- **Quantum Circuit**: 8-qubit variational circuit with strongly-entangling layers
- **Classical Postprocessing**: Dense neural network layers for final classification

## Notebooks

### 1. `01_vqc_nsl_kdd.ipynb` - Training
Main training notebook for VQC models on NSL-KDD dataset.

**Features:**
- Data loading and preprocessing
- PCA dimensionality reduction
- Hybrid quantum-classical model definition
- Complete training loop with early stopping
- Model checkpointing and saving
- Comprehensive evaluation and visualization

**Outputs:**
- Trained model: `results/models/quantum/vqc_hybrid_nsl_kdd_*.pt`
- Preprocessing pipeline: `results/models/quantum/vqc_preprocessing_*.pkl`
- Training history: `results/logs/vqc_hybrid_history_*.json`
- Metrics: `results/logs/vqc_metrics_*.json`
- Visualizations: `results/figures/vqc_*.png`

### 2. `02_vqc_inference.ipynb` - Inference & Analysis
Comprehensive evaluation and analysis of trained VQC models.

**Features:**
- Load trained VQC models
- Test set evaluation
- Detailed performance metrics
- Error analysis
- Model comparison with classical baselines
- Extensive visualizations

**Outputs:**
- Inference results: `results/logs/vqc_inference_results_*.json`
- Analysis figures: `results/figures/vqc_inference_*.png`

## Requirements

Make sure you have the quantum dependencies installed:

```bash
pip install -r requirements-quantum.txt
```

Key packages:
- `pennylane>=0.34.0`
- `pennylane-lightning>=0.34.0`
- `torch>=2.1.0`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`

## Quick Start

### 1. Train VQC Model

Open and run `01_vqc_nsl_kdd.ipynb`:

```python
# The notebook will:
# 1. Load NSL-KDD dataset
# 2. Apply PCA to reduce dimensions to 8
# 3. Train hybrid quantum-classical classifier
# 4. Save model and results
```

**Training Configuration:**
- Qubits: 8
- Quantum Layers: 4
- Batch Size: 32
- Epochs: 50 (with early stopping)
- Learning Rate: 0.001

**Note:** Training is slower than classical models due to quantum circuit simulation (~30-60 minutes for 10K samples).

### 2. Evaluate Model

Open and run `02_vqc_inference.ipynb`:

```python
# The notebook will:
# 1. Load the trained VQC model
# 2. Evaluate on test set
# 3. Generate detailed metrics and visualizations
# 4. Compare with classical baselines
```

## Model Architecture

### Hybrid Quantum Classifier

```
Input (122 features)
    ↓
StandardScaler (normalization)
    ↓
PCA Reduction (→ 8 features)
    ↓
[8-Qubit Quantum Circuit]
  │
  ├─ Angle Encoding (RY gates)
  ├─ Strongly Entangling Layers (4 layers)
  ├─ Pauli-Z Measurements
  │
  └─ Output (8 quantum features)
    ↓
[Classical Postprocessing]
  │
  ├─ Dense(8 → 16) + ReLU + Dropout
  ├─ Dense(16 → 8) + ReLU + Dropout
  └─ Dense(8 → 2)
    ↓
Output (2 classes: Normal/Attack)
```

## Performance Expectations

Based on the hybrid approach:
- **Expected Accuracy**: 80-90%
- **Training Time**: 30-60 minutes (10K samples)
- **Inference**: ~0.1-0.5 seconds per sample

### Trade-offs

**Advantages:**
- ✅ Quantum circuit can capture complex non-linear patterns
- ✅ Hybrid approach leverages both quantum and classical strengths
- ✅ PCA ensures efficient qubit usage

**Limitations:**
- ⚠️ Significantly slower than classical models
- ⚠️ Limited by number of qubits (8)
- ⚠️ PCA dimensionality reduction may lose information
- ⚠️ Quantum simulation is CPU-bound

## Customization

### Adjust Qubit Count

In `01_vqc_nsl_kdd.ipynb`, modify:

```python
N_QUBITS = 8  # Change to 4, 12, or 16
N_QUANTUM_LAYERS = 4  # Adjust circuit depth
```

**Note:** More qubits = longer training time but potentially better performance.

### Use Subset for Quick Testing

For faster experimentation:

```python
USE_SUBSET = True  # Enable subset training
SUBSET_SIZE = 5000  # Use 5K samples instead of full dataset
```

### Change Quantum Device

```python
QUANTUM_DEVICE = 'lightning.qubit'  # Faster simulator
# or
QUANTUM_DEVICE = 'default.qubit'    # Default simulator
```

## Troubleshooting

### PennyLane Not Found

```bash
pip install pennylane pennylane-lightning
```

### CUDA Out of Memory

Quantum circuits run on CPU, but classical layers use GPU:
```python
BATCH_SIZE = 16  # Reduce batch size
```

### Training Too Slow

```python
USE_SUBSET = True
SUBSET_SIZE = 5000  # Use smaller training set
N_QUANTUM_LAYERS = 2  # Reduce circuit depth
```

## References

- **PennyLane Documentation**: https://pennylane.ai/
- **Variational Quantum Classifiers**: https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
- **NSL-KDD Dataset**: https://www.unb.ca/cic/datasets/nsl.html

## Next Steps

1. ✅ Run `01_vqc_nsl_kdd.ipynb` to train the model
2. ✅ Run `02_vqc_inference.ipynb` for evaluation
3. 📊 Compare with classical baselines (CNN, LSTM, Transformer)
4. 🔬 Experiment with different configurations (qubits, layers, encoding)
5. 📝 Update project documentation with results

---

**For questions or issues, refer to the main project README or walkthrough documentation.**
