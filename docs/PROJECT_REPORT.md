# NIDS-DL Project Report
## Network Intrusion Detection using Deep Learning

**Project Date:** January 2026  
**Team Members:** [Your Names Here]

---

## Executive Summary

This project implements and evaluates deep learning models for Network Intrusion Detection Systems (NIDS). We trained and tested **LSTM** and **CNN** models on two benchmark datasets: **NSL-KDD** and **UNSW-NB15**. The best performing model achieved **88.78% test accuracy** using CNN on UNSW-NB15.

---

## 1. Datasets

### 1.1 NSL-KDD
| Property | Value |
|----------|-------|
| Training Samples | 125,973 |
| Testing Samples | 22,544 |
| Features | 41 |
| Classification | Binary (Normal/Attack) |
| Attack Types | 39 categories mapped to binary |

### 1.2 UNSW-NB15
| Property | Value |
|----------|-------|
| Training Samples | 175,341 |
| Testing Samples | 82,332 |
| Features | 42 (after preprocessing) |
| Classification | Binary (Normal/Attack) |
| Attack Categories | 9 types |

---

## 2. Model Architectures

### 2.1 LSTM (Long Short-Term Memory)
- **Architecture:** Bidirectional LSTM with attention
- **Layers:** 3 LSTM layers, 256 hidden units
- **Regularization:** Dropout (0.4), BatchNorm, Gradient Clipping
- **Optimizer:** AdamW with LR scheduling

### 2.2 CNN (Convolutional Neural Network)
- **Architecture:** 1D Convolutional layers
- **Filters:** [64, 128, 256] with kernel size 3
- **Pooling:** MaxPool1d
- **Regularization:** Dropout (0.4), BatchNorm

---

## 3. Experimental Results

### 3.1 Performance Summary

| Dataset | Model | Test Acc | Val Acc | Precision | Recall | F1-Score |
|---------|-------|----------|---------|-----------|--------|----------|
| NSL-KDD | LSTM | **80.78%** | 99.68% | 0.85 | 0.81 | 0.81 |
| NSL-KDD | CNN | 78.84% | 99.60% | 0.85 | 0.79 | 0.79 |
| UNSW-NB15 | LSTM | 88.48% | 96.90% | 0.91 | 0.88 | 0.89 |
| UNSW-NB15 | CNN | **88.78%** | 95.87% | 0.91 | 0.89 | 0.89 |

### 3.2 Best Model: CNN on UNSW-NB15

```
Test Accuracy: 88.78%
Validation Accuracy: 95.87%

              precision    recall  f1-score   support

      Normal       0.75      0.98      0.85     56000
      Attack       0.99      0.84      0.91    119341

    accuracy                           0.89    175341

Confusion Matrix:
TN: 54,863 | FP: 1,137
FN: 18,535 | TP: 100,806
```

### 3.3 NSL-KDD Results (LSTM)

```
Test Accuracy: 80.78%
Validation Accuracy: 99.68%

              precision    recall  f1-score   support

      Normal       0.70      0.98      0.81      9711
      Attack       0.97      0.68      0.80     12833

    accuracy                           0.81     22544
```

---

## 4. Key Findings

### 4.1 Dataset Comparison
- **UNSW-NB15** shows better generalization (smaller val-test gap: ~7% vs ~19%)
- NSL-KDD exhibits significant overfitting despite regularization
- UNSW-NB15 provides more realistic network traffic patterns

### 4.2 Model Comparison
- **CNN** slightly outperforms LSTM on UNSW-NB15 (88.78% vs 88.48%)
- **LSTM** performs better on NSL-KDD (80.78% vs 78.84%)
- Both models achieve high attack precision (97-99%)

### 4.3 Detection Analysis
- **Normal traffic detection:** 98% recall (low false negatives)
- **Attack detection:** 84% recall with 99% precision
- Trade-off: Prioritizing low false alarms over catching all attacks

---

## 5. Project Structure

```
NIDS-DL/
├── configs/                    # Configuration files
├── data/raw/                   # Dataset storage
│   ├── nsl-kdd/               # NSL-KDD dataset
│   └── unsw-nb15/             # UNSW-NB15 dataset
├── results/
│   ├── models/                # Saved model weights
│   │   ├── lstm_improved_best.pt
│   │   ├── cnn_best.pt
│   │   ├── lstm_unsw_nb15_best.pt
│   │   └── cnn_unsw_nb15_best.pt
│   ├── lstm_improved_results.txt
│   ├── cnn_results.txt
│   ├── lstm_unsw_nb15_results.txt
│   └── cnn_unsw_nb15_results.txt
├── src/                       # Source code modules
│   ├── models/               # Model implementations
│   ├── data/                 # Data loading
│   └── training/             # Training utilities
├── train_lstm_improved.py    # NSL-KDD LSTM training
├── train_cnn.py              # NSL-KDD CNN training
├── train_lstm_unsw.py        # UNSW-NB15 LSTM training
└── train_cnn_unsw.py         # UNSW-NB15 CNN training
```

---

## 6. Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 512 |
| Max Epochs | 100 |
| Learning Rate | 0.001 (LSTM: 0.0005) |
| Weight Decay | 1e-4 |
| Early Stopping Patience | 15-20 epochs |
| Optimizer | AdamW |
| LR Scheduler | ReduceLROnPlateau |
| Class Weighting | Inverse frequency |

---

## 7. Technologies Used

- **Deep Learning:** PyTorch 2.1+
- **CUDA:** GPU acceleration (NVIDIA)
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Rich console output
- **Python:** 3.12

---

## 8. Future Work

1. **Transformer Model** - Implement self-attention for complex patterns
2. **Autoencoder** - Unsupervised anomaly detection
3. **Quantum ML** - PennyLane variational quantum classifiers
4. **Multi-class Classification** - Detect specific attack types
5. **Real-time Inference** - Deploy models for live detection

---

## 9. Conclusion

This project successfully demonstrates the application of deep learning to network intrusion detection. The CNN model on UNSW-NB15 achieved the best test accuracy of **88.78%** with high attack detection precision (99%). The project provides a solid foundation for further research into more advanced architectures and real-world deployment.

---

## 10. References

1. NSL-KDD Dataset - https://www.unb.ca/cic/datasets/nsl.html
2. UNSW-NB15 Dataset - https://research.unsw.edu.au/projects/unsw-nb15-dataset
3. PyTorch Documentation - https://pytorch.org/docs/

---

*Report Generated: January 4, 2026*
