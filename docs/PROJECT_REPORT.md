# NIDS-DL Project Report
## Network Intrusion Detection using Deep Learning

**Project Date:** December 2025 – March 2026  
**Team Members:** [Your Names Here]  
**Status:** 85% Complete (3rd Review)

---

## Executive Summary

This project implements and evaluates deep learning models for Network Intrusion Detection Systems (NIDS). We trained and tested **CNN**, **LSTM**, **Transformer**, and **Autoencoder** models across four benchmark datasets: **NSL-KDD**, **UNSW-NB15**, **CICIDS2017**, and **CICIDS2018**. Additionally, we explored **Variational Quantum Classifiers (VQC)** using PennyLane for hybrid quantum-classical intrusion detection. A production-ready **Streamlit Pro Dashboard** with live traffic analysis and batch processing capabilities has been developed. The best performing model achieved **98.00% test accuracy** using CNN on CICIDS2017.

---

## 1. Introduction

### 1.1 Problem Statement
Traditional Intrusion Detection Systems (IDS) rely on fixed signatures and rule-based approaches, which fail to detect zero-day attacks and sophisticated threat variants. Deep learning offers the ability to automatically learn complex, non-linear patterns in network traffic data, enabling detection of previously unseen attack vectors.

### 1.2 Objectives
1. Implement and compare four deep learning architectures (CNN, LSTM, Transformer, Autoencoder) for binary network intrusion detection.
2. Evaluate model performance across four benchmark datasets of increasing scale and complexity.
3. Explore quantum machine learning for potential performance gains in pattern recognition.
4. Develop a real-time threat detection dashboard for practical deployment.

### 1.3 Scope
- **Binary classification:** Normal vs. Attack traffic
- **Four datasets:** NSL-KDD, UNSW-NB15, CICIDS2017, CICIDS2018
- **Four architectures:** CNN, LSTM, Transformer, Autoencoder
- **Quantum ML:** Variational Quantum Classifiers (VQC) using PennyLane
- **Application:** Real-time and batch threat analysis dashboard

---

## 2. Datasets

### 2.1 NSL-KDD
| Property | Value |
|----------|-------|
| Training Samples | 125,973 |
| Testing Samples | 22,544 |
| Features | 41 (122 after encoding) |
| Classification | Binary (Normal/Attack) |
| Attack Types | 39 categories mapped to binary |

### 2.2 UNSW-NB15
| Property | Value |
|----------|-------|
| Training Samples | 175,341 |
| Testing Samples | 82,332 |
| Features | 42 (after preprocessing) |
| Classification | Binary (Normal/Attack) |
| Attack Categories | 9 types |

### 2.3 CICIDS2017
| Property | Value |
|----------|-------|
| Total Samples | 2,830,743 |
| After Cleaning | 2,827,876 |
| Features | 77 |
| Classification | Binary (Benign/Attack) |
| Benign Samples | 2,271,320 (80.3%) |
| Attack Samples | 556,556 (19.7%) |
| Attack Types | DDoS, Port Scan, Web Attacks, Infiltration, Bot, FTP/SSH Patator |

### 2.4 CICIDS2018
| Property | Value |
|----------|-------|
| Total Samples | 8,284,254 |
| After Cleaning | ~8,247,888 |
| Features | 76 |
| Classification | Binary (Benign/Attack) |
| Benign Samples | 6,077,145 (73.4%) |
| Attack Samples | 2,170,743 (26.6%) |

---

## 3. Model Architectures

### 3.1 CNN (Convolutional Neural Network)
- **Architecture:** 1D Convolutional layers with residual connections
- **Filters:** [64, 128, 256] with kernel size 3
- **Pooling:** AdaptiveAvgPool1d / MaxPool1d
- **Regularization:** Dropout (0.3–0.4), BatchNorm
- **Loss:** Focal Loss (α=0.25, γ=2) for class imbalance handling

### 3.2 LSTM (Long Short-Term Memory)
- **Architecture:** Bidirectional LSTM with attention mechanism
- **Layers:** 3 LSTM layers, 256 hidden units
- **Regularization:** Dropout (0.4), BatchNorm, Gradient Clipping
- **Optimizer:** AdamW with LR scheduling

### 3.3 Transformer (Self-Attention)
- **Architecture:** Multi-head self-attention encoder with Pre-LayerNorm
- **Embedding:** 64-dim, 4 attention heads
- **Layers:** 3 transformer blocks with feed-forward (128 units)
- **Pooling:** Global average pooling from sequence
- **Regularization:** Dropout (0.3), Label Smoothing

### 3.4 Autoencoder (Anomaly Detection)
- **Architecture:** Encoder-Decoder with bottleneck
- **Encoder:** Input → 64 → 32 → 16 → 8 (latent)
- **Decoder:** 8 → 16 → 32 → 64 → Input
- **Training Modes:**
  - **Unsupervised:** Normal traffic only; high reconstruction error = anomaly
  - **Supervised (SAE):** Classification head + reconstruction loss for CICIDS2018

### 3.5 Variational Quantum Classifier (VQC) – Exploratory
- **Architecture:** Hybrid quantum-classical model
- **Preprocessing:** PCA dimensionality reduction (122 → 8 features)
- **Quantum Circuit:** 8-qubit variational circuit with strongly-entangling layers (4 layers)
- **Encoding:** Angle Encoding (RY gates)
- **Measurements:** Pauli-Z on all qubits
- **Classical Postprocessing:** Dense(8→16) → ReLU → Dense(16→8) → ReLU → Dense(8→2)
- **Framework:** PennyLane with `lightning.qubit` simulator

---

## 4. Experimental Results

### 4.1 Complete Performance Summary

| Dataset | Model | Test Acc | Precision | Recall | F1-Score | Notes |
|---------|-------|----------|-----------|--------|----------|-------|
| **NSL-KDD** | LSTM (Improved) | **81.12%** | 0.86 | 0.81 | 0.81 | Threshold=0.30 |
| NSL-KDD | CNN (Improved) | 78.69% | 0.84 | 0.79 | 0.78 | Threshold=0.30 |
| NSL-KDD | Transformer (Improved) | 78.04% | 0.84 | 0.78 | 0.78 | Pre-LN + Warmup |
| NSL-KDD | Autoencoder | 85.53% | 0.86 | 0.86 | 0.87 | ROC-AUC: 0.9468 |
| **UNSW-NB15** | CNN (Improved) | **94.23%** | 0.94 | 0.94 | 0.94 | Class Weights |
| UNSW-NB15 | Transformer (Improved) | 93.22% | 0.93 | 0.93 | 0.93 | Threshold=0.45 |
| UNSW-NB15 | LSTM | 88.48% | 0.91 | 0.88 | 0.89 | — |
| UNSW-NB15 | CNN (Original) | 88.78% | 0.91 | 0.89 | 0.89 | — |
| UNSW-NB15 | Autoencoder | 88.04% | 0.91 | 0.88 | 0.88 | ROC-AUC: 0.9809 |
| **CICIDS2017** | CNN | **98.00%** | 0.97 | 0.97 | 0.97 | Best overall |
| CICIDS2017 | Transformer | ~97.50% | 0.98 | 0.97 | 0.98 | — |
| CICIDS2017 | LSTM | ~97.00% | 0.98 | 0.97 | 0.97 | — |
| **CICIDS2018** | CNN | **96.43%** | 0.97 | 0.96 | 0.96 | — |
| CICIDS2018 | Autoencoder (SAE) | 96.16% | 0.96 | 0.96 | 0.96 | Supervised |
| CICIDS2018 | Transformer | 96.05% | 0.96 | 0.96 | 0.96 | — |
| CICIDS2018 | LSTM | 95.90% | 0.96 | 0.96 | 0.96 | — |

### 4.2 Best Model per Dataset

| Dataset | Best Model | Accuracy | Key Strength |
|---------|------------|----------|--------------|
| NSL-KDD | Autoencoder | 85.53% | Unsupervised anomaly detection |
| UNSW-NB15 | CNN (Improved) | 94.23% | +5.45% over original with class weights |
| CICIDS2017 | CNN | 98.00% | Best overall accuracy |
| CICIDS2018 | CNN | 96.43% | Consistent top performer |

### 4.3 Detailed Results – Best Performers

#### CNN on CICIDS2017 (Best Overall)
```
Test Accuracy: 98.00%

              precision    recall  f1-score   support

      Benign       0.99      0.99      0.99    340698
      Attack       0.95      0.95      0.95     83484

    accuracy                           0.98    424182
```

#### CNN on CICIDS2018
```
Test Accuracy: 96.43%

              precision    recall  f1-score   support

      Benign       0.94      0.99      0.97     75000
      Attack       0.99      0.94      0.96     75000

    accuracy                           0.96    150000

Confusion Matrix:
TN: 74,514 | FP: 486
FN: 4,863 | TP: 70,137
```

#### CNN (Improved) on UNSW-NB15
```
Test Accuracy: 94.23%

              precision    recall  f1-score   support

      Normal       0.91      0.93      0.92      9300
      Attack       0.96      0.95      0.95     16468

    accuracy                           0.94     25768

Confusion Matrix:
TN: 8,621 | FP: 679
FN: 808 | TP: 15,660
```

#### Autoencoder (Supervised) on CICIDS2018
```
Test Accuracy: 96.16%

              precision    recall  f1-score   support

      Benign       0.94      0.99      0.96     30000
      Attack       0.99      0.93      0.96     30000

    accuracy                           0.96     60000

Confusion Matrix:
TN: 29,755 | FP: 245
FN: 2,062  | TP: 27,938
```
> **Note**: Switching the autoencoder to supervised mode (classification + reconstruction loss) resolved an accuracy bottleneck, improving from 71% to 96.16%.

---

## 5. Key Findings & Analysis

### 5.1 Dataset Comparison
- **CICIDS datasets** show the best generalization with 96–98% accuracy, reflecting modern and realistic network traffic patterns.
- **UNSW-NB15 (Improved)** saw a significant jump from 88.78% → **94.23%** with class-weighted loss and threshold tuning.
- **NSL-KDD** exhibits significant overfitting (val ~99% vs test ~78–81%) despite regularization, due to train-test distribution mismatch.

### 5.2 Model Comparison
- **CNN** is the most consistent top performer across all four datasets.
- **LSTM** captures temporal/sequential dependencies effectively on larger datasets.
- **Transformer** excels on CICIDS2017 (~97.5%) but is computationally more expensive.
- **Autoencoder (Unsupervised)** provides valuable zero-day detection capability via reconstruction error.

### 5.3 Improvements Applied (Post-2nd Review)
| Technique | Impact |
|-----------|--------|
| **Class-Weighted Loss** | Fixed class imbalance; improved UNSW-NB15 by +5.45% |
| **Threshold Tuning** | Optimized decision boundary for better precision-recall tradeoff |
| **Label Smoothing** | Reduced overconfident predictions and overfitting |
| **Pre-LayerNorm** (Transformer) | Stabilized training and improved convergence |
| **Focal Loss** (CICIDS) | Handled 80:20 class imbalance effectively |
| **Supervised Autoencoder** | Resolved CICIDS2018 autoencoder bottleneck (71% → 96%) |

---

## 6. Quantum Machine Learning

### 6.1 Approach
We implemented a **Hybrid Quantum-Classical** model using PennyLane:
1. **Classical Preprocessing:** PCA reduces 122 features → 8 features
2. **Quantum Circuit:** 8-qubit variational circuit with 4 strongly-entangling layers
3. **Classical Postprocessing:** Dense layers for final classification

### 6.2 Configuration
| Parameter | Value |
|-----------|-------|
| Qubits | 8 |
| Quantum Layers | 4 |
| Encoding | Angle Encoding (RY gates) |
| Simulator | PennyLane `lightning.qubit` |
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Expected Accuracy | 80–90% |

### 6.3 Datasets Applied
- **NSL-KDD:** Full training and inference notebooks (`01_vqc_nsl_kdd.ipynb`, `02_vqc_inference.ipynb`)
- **CICIDS2018:** VQC implementation notebook (`03_vqc_cicids2018.ipynb`)

### 6.4 Trade-offs
| Aspect | Details |
|--------|---------|
| ✅ Strengths | Captures complex non-linear patterns; hybrid approach leverages quantum + classical |
| ⚠️ Limitations | Significantly slower (~30–60 min for 10K samples); limited by 8 qubits; PCA may lose information |

---

## 7. Application Development

### 7.1 Pro Dashboard (Streamlit)
A premium, production-ready dashboard (`frontend/pro_dashboard.py`) was developed with:
- **Glassmorphism UI** with dark mode styling
- **Live Threat Analysis:** Real-time network packet classification
- **Batch Processing:** Upload CSV/PCAP files for bulk analysis
- **Visualizations:** Attack distribution charts, threat level indicators, model performance metrics
- **Model Selection:** Choose between CNN, LSTM, Transformer, and Autoencoder models

### 7.2 Live Network Sniffer
A standalone live traffic analysis script (`scripts/live_sniffer.py`) that:
- Captures live network packets
- Extracts 77 features matching the CICIDS training format
- Feeds features into trained models for instant classification
- Provides real-time Normal/Attack predictions

### 7.3 Full-Stack Architecture
- **Backend:** FastAPI-based API layer (`backend/`) for model inference
- **Frontend:** React + Vite application (`frontend/`) for enterprise-grade UI
- **Streamlit App:** Standalone dashboard (`streamlit_app/`) for quick deployment

---

## 8. Project Structure

```
NIDS-DL/
├── configs/                        # Configuration files
├── data/raw/                       # Dataset storage
│   ├── nsl-kdd/                   # NSL-KDD dataset
│   ├── unsw-nb15/                 # UNSW-NB15 dataset
│   ├── cicids2017/                # CICIDS2017 (8 CSV files)
│   └── cicids2018/                # CICIDS2018 dataset
├── notebooks/
│   ├── 01_data_exploration/       # EDA notebooks per dataset
│   ├── 02_classical_dl/           # CNN, LSTM, Transformer, AE training
│   ├── 03_quantum_dl/            # VQC notebooks (NSL-KDD, CICIDS2018)
│   └── 04_visualization/         # Hilbert curve & visualization experiments
├── results/
│   ├── models/                    # 36+ saved model weights (.pt/.pth)
│   │   ├── best_cnn_cicids2017.pth
│   │   ├── best_lstm_cicids2017.pth
│   │   ├── best_transformer_cicids2017.pth
│   │   ├── best_cnn_cicids2018.pth
│   │   └── ... (all datasets × all models)
│   ├── *_results.txt              # Classification reports per model
│   └── figures/                   # Training plots and visualizations
├── scripts/
│   ├── train_*.py                 # Standalone training scripts (all datasets)
│   ├── live_sniffer.py            # Real-time network traffic analyzer
│   ├── generate_*_scaler.py       # Feature scaler generation
│   └── verify_setup.py            # GPU/CUDA verification
├── frontend/
│   ├── pro_dashboard.py           # Streamlit Pro Dashboard
│   └── src/                       # React frontend application
├── backend/                        # FastAPI backend
├── streamlit_app/                  # Standalone Streamlit deployment
├── src/
│   ├── models/                    # Model architecture definitions
│   ├── data/                      # Data loading & preprocessing
│   └── training/                  # Training utilities
├── docs/
│   ├── PROJECT_REPORT.md          # This report
│   ├── MODEL_COMPARISON.md        # Detailed model comparisons
│   ├── literature_survey.md       # Literature review
│   └── gantt_chart.html           # Project timeline
└── tests/                          # Unit tests
```

---

## 9. Training Configuration

| Parameter | NSL-KDD / UNSW-NB15 | CICIDS2017 / 2018 |
|-----------|---------------------|-------------------|
| Batch Size | 512 | 1024 |
| Max Epochs | 100 | 20–50 |
| Learning Rate | 0.001 (LSTM: 0.0005) | 0.001 |
| Weight Decay | 1e-4 | 1e-4 |
| Early Stopping | 15–20 epochs | 5–10 epochs |
| Optimizer | AdamW | Adam / AdamW |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Class Weighting | Inverse frequency | Focal Loss |
| GPU | NVIDIA GeForce RTX 4050 | NVIDIA GeForce RTX 4050 |

---

## 10. Technologies Used

| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch 2.1+ |
| Quantum ML | PennyLane 0.34+ |
| GPU Acceleration | CUDA (NVIDIA RTX 4050) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Dashboard | Streamlit |
| Frontend | React, Vite |
| Backend | FastAPI |
| Visualization | Matplotlib, Seaborn, Plotly |
| Version Control | Git, Git LFS (model weights) |
| Python | 3.12 |

---

## 11. Timeline & Progress

| Phase | Timeline | Milestone | Status |
|-------|----------|-----------|--------|
| 1 | Dec 2025 – Jan 2026 | Literature Survey & System Design | ✅ Complete |
| 2 | Jan 2026 | NSL-KDD & UNSW-NB15 (30% Review) | ✅ Complete |
| 3 | Jan – Feb 7, 2026 | CICIDS2017 (50% / 2nd Review) | ✅ Complete |
| 4 | Feb 7 – Mar 4, 2026 | CICIDS2018 + Improvements (3rd Review) | ✅ Complete |
| 5 | Feb – Mar 2026 | Quantum ML (VQC) | ✅ 90% |
| 6 | Feb – Mar 2026 | Pro Dashboard & Live Sniffer | 🟢 80% |
| 7 | Mar – Apr 2026 | Final Integration, Report & Submission | 🔜 Upcoming |

---

## 12. Future Work

1. ~~**Transformer Model** – Implement self-attention for complex patterns~~ ✅ COMPLETED
2. ~~**Autoencoder** – Unsupervised anomaly detection~~ ✅ COMPLETED
3. ~~**CICIDS Datasets** – Train on CICIDS2017 and CICIDS2018~~ ✅ COMPLETED
4. ~~**Quantum ML** – PennyLane variational quantum classifiers~~ ✅ COMPLETED
5. **Multi-class Classification** – Detect specific attack types (DDoS, Probe, etc.)
6. **Ensemble Methods** – Combine CNN, LSTM, and Autoencoder outputs via voting
7. **Real-time Optimization** – Optimize inference pipeline for high-bandwidth networks
8. **Variational Autoencoder (VAE)** – Better anomaly scoring with latent distributions
9. **Production Deployment** – Containerize and deploy with Docker

---

## 13. Conclusion

This project successfully demonstrates the application of deep learning and exploratory quantum machine learning to network intrusion detection. Key achievements include:

- **98.00% accuracy** with CNN on CICIDS2017 (best overall)
- **96.43% accuracy** with CNN on CICIDS2018 (8.2M+ samples)
- **94.23% accuracy** on UNSW-NB15 with improved training techniques (+5.45% over baseline)
- **96.16% accuracy** with supervised autoencoder on CICIDS2018, resolving a performance bottleneck
- **Hybrid Quantum-Classical** VQC implementation using PennyLane
- **Production-ready dashboard** with live threat analysis and batch processing

All four model architectures demonstrate high attack detection precision (95–99%) across modern datasets, providing a solid foundation for real-world deployment and further research into advanced architectures.

---

## 14. References

1. NSL-KDD Dataset – https://www.unb.ca/cic/datasets/nsl.html
2. UNSW-NB15 Dataset – https://research.unsw.edu.au/projects/unsw-nb15-dataset
3. CICIDS2017 Dataset – https://www.unb.ca/cic/datasets/ids-2017.html
4. CICIDS2018 Dataset – https://www.unb.ca/cic/datasets/ids-2018.html
5. PyTorch Documentation – https://pytorch.org/docs/
6. PennyLane Documentation – https://pennylane.ai/
7. PennyLane VQC Tutorial – https://pennylane.ai/qml/demos/tutorial_variational_classifier.html

---

*Report Updated: March 4, 2026*  
*Project: NIDS-DL – Deep Learning for Network Security*
