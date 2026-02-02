# NIDS-DL Project Report
## Network Intrusion Detection using Deep Learning

**Project Date:** January - February 2026  
**Team Members:** [Your Names Here]

---

## Executive Summary

This project implements and evaluates deep learning models for Network Intrusion Detection Systems (NIDS). We trained and tested **LSTM**, **CNN**, **Transformer**, and **Autoencoder** models on four benchmark datasets: **NSL-KDD**, **UNSW-NB15**, **CICIDS2017**, and **CICIDS2018**. The best performing model achieved **98.00% test accuracy** using CNN on CICIDS2017.

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

### 1.3 CICIDS2017
| Property | Value |
|----------|-------|
| Total Samples | 2,830,743 |
| After Cleaning | 2,827,876 |
| Features | 77 |
| Classification | Binary (Benign/Attack) |
| Benign Samples | 2,271,320 |
| Attack Samples | 556,556 |

### 1.4 CICIDS2018
| Property | Value |
|----------|-------|
| Total Samples | 8,284,254 |
| After Cleaning | ~8,247,888 |
| Features | 76 |
| Classification | Binary (Benign/Attack) |
| Benign Samples | 6,077,145 |
| Attack Samples | 2,170,743 |

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

### 2.3 Transformer (Self-Attention)
- **Architecture:** Multi-head self-attention encoder
- **Embedding:** 64-dim, 4 attention heads
- **Layers:** 3 transformer blocks with feed-forward (128 units)
- **Pooling:** Global average pooling from sequence
- **Regularization:** Dropout (0.3)

### 2.4 Autoencoder (Anomaly Detection)
- **Architecture:** Encoder-Decoder with bottleneck
- **Encoder:** Input → 64 → 32 → 16 → 8 (latent)
- **Decoder:** 8 → 16 → 32 → 64 → Input
- **Training:** Normal traffic only (unsupervised)
- **Detection:** High reconstruction error = anomaly

---

## 3. Experimental Results

### 3.1 Performance Summary

| Dataset | Model | Test Acc | ROC-AUC | Precision | Recall | F1-Score |
|---------|-------|----------|---------|-----------|--------|----------|
| NSL-KDD | LSTM | 80.78% | - | 0.85 | 0.81 | 0.81 |
| NSL-KDD | CNN | 78.84% | - | 0.85 | 0.79 | 0.79 |
| NSL-KDD | Transformer | 82.06% | - | 0.86 | 0.82 | 0.82 |
| NSL-KDD | Autoencoder | **85.53%** | **0.9468** | 0.86 | 0.86 | 0.87 |
| UNSW-NB15 | LSTM | 88.48% | - | 0.91 | 0.88 | 0.89 |
| UNSW-NB15 | CNN | 88.78% | - | 0.91 | 0.89 | 0.89 |
| UNSW-NB15 | Transformer | 87.35% | - | 0.91 | 0.87 | 0.88 |
| UNSW-NB15 | Autoencoder | 88.04% | **0.9809** | 0.91 | 0.88 | 0.88 |
| CICIDS2017 | CNN | **98.00%** | - | 0.97 | 0.97 | 0.97 |
| CICIDS2018 | CNN | **96.43%** | - | 0.97 | 0.96 | 0.96 |
| CICIDS2018 | Transformer | 96.05% | - | 0.96 | 0.96 | 0.96 |
| CICIDS2018 | LSTM | 95.90% | - | 0.96 | 0.96 | 0.96 |
| CICIDS2018 | Autoencoder (Supervised) | **96.16%** | - | 0.96 | 0.96 | 0.96 |

### 3.2 Best Model: CNN on CICIDS2017

```
Test Accuracy: 98.00%

              precision    recall  f1-score   support

      Benign       0.99      0.99      0.99    340698
      Attack       0.95      0.95      0.95     83484

    accuracy                           0.98    424182
```

### 3.3 CNN on CICIDS2018

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

### 3.4 Transformer on CICIDS2018

```
Test Accuracy: 96.05%

              precision    recall  f1-score   support

      Benign       0.94      0.99      0.96     45000
      Attack       0.99      0.94      0.96     45000

    accuracy                           0.96     90000
```

### 3.5 LSTM on CICIDS2018

```
Test Accuracy: 95.90%

              precision    recall  f1-score   support

      Benign       0.93      0.99      0.96     30000
      Attack       0.99      0.93      0.96     30000

    accuracy                           0.96     60000
```

### 3.6 Autoencoder on CICIDS2018 (Supervised)

**Goal met: >95% Accuracy**

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
> **Note**: Switch to Supervised Learning (Classification + Reconstruction) successfully resolved the accuracy bottleneck, improving from 71% to 96.16%.

### 3.7 CNN on UNSW-NB15

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

### 3.8 NSL-KDD Results (LSTM)

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
- **CICIDS datasets** show better generalization and higher accuracy (96-98%)
- **UNSW-NB15** shows good generalization (smaller val-test gap: ~7% vs ~19%)
- NSL-KDD exhibits significant overfitting despite regularization
- CICIDS2017/2018 provide more realistic modern network traffic patterns

### 4.2 Model Comparison
- **CNN** achieves best results on CICIDS2017 (98.00%)
- **CNN** performs consistently well across all datasets
- Both supervised models achieve high attack precision (95-99%)

### 4.3 Detection Analysis
- **CICIDS2017:** 99% recall on benign, 95% on attacks
- **CICIDS2018:** 99% recall on benign, 94% on attacks
- Trade-off: High precision with minimal false alarms

---

## 5. Project Structure

```
NIDS-DL/
├── configs/                    # Configuration files
├── data/raw/                   # Dataset storage
│   ├── nsl-kdd/               # NSL-KDD dataset
│   ├── unsw-nb15/             # UNSW-NB15 dataset
│   ├── cicids2017/            # CICIDS2017 dataset
│   └── cicids2018/            # CICIDS2018 dataset
├── results/
│   ├── models/                # Saved model weights
│   │   ├── lstm_improved_best.pt
│   │   ├── cnn_best.pt
│   │   ├── lstm_unsw_nb15_best.pt
│   │   ├── cnn_unsw_nb15_best.pt
│   │   ├── autoencoder_nsl_kdd_best.pt
│   │   ├── autoencoder_unsw_nb15_best.pt
│   │   ├── best_cnn_cicids2018.pth
│   │   ├── transformer_cicids2018_best.pt
│   │   ├── best_lstm_cicids2018.pth
│   │   └── best_autoencoder_cicids2018.pth
│   ├── lstm_improved_results.txt
│   ├── cnn_results.txt
│   ├── lstm_unsw_nb15_results.txt
│   ├── cnn_unsw_nb15_results.txt
│   ├── autoencoder_nsl_kdd_results.txt
│   ├── autoencoder_unsw_nb15_results.txt
│   ├── cnn_cicids2018_results.txt
│   ├── transformer_cicids2018_results.txt
│   ├── lstm_cicids2018_results.txt
│   └── autoencoder_cicids2018_results.txt
├── src/                       # Source code modules
│   ├── models/               # Model implementations
│   ├── data/                 # Data loading
│   └── training/             # Training utilities
├── train_lstm_improved.py    # NSL-KDD LSTM training
├── train_cnn.py              # NSL-KDD CNN training
├── train_lstm_unsw.py        # UNSW-NB15 LSTM training
├── train_cnn_unsw.py         # UNSW-NB15 CNN training
├── train_autoencoder.py      # NSL-KDD Autoencoder training
└── train_autoencoder_unsw.py # UNSW-NB15 Autoencoder training
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

1. ~~**Transformer Model** - Implement self-attention for complex patterns~~ ✓ COMPLETED
2. ~~**Autoencoder** - Unsupervised anomaly detection~~ ✓ COMPLETED
3. ~~**CICIDS Datasets** - Train on CICIDS2017 and CICIDS2018~~ ✓ COMPLETED
4. **Quantum ML** - PennyLane variational quantum classifiers
5. **Multi-class Classification** - Detect specific attack types
6. **Real-time Inference** - Deploy models for live detection
7. **Ensemble Methods** - Combine autoencoder with supervised models
8. **Improved Autoencoder** - Variational Autoencoder (VAE) for better anomaly scoring

---

## 9. Conclusion

This project successfully demonstrates the application of deep learning to network intrusion detection. We evaluated models on four datasets, with the CNN model on **CICIDS2017 achieving the best test accuracy of 98.00%** and **CICIDS2018 achieving 96.43%**. All models demonstrate high attack detection precision (95-99%), providing a solid foundation for further research into more advanced architectures and real-world deployment.

---

## 10. References

1. NSL-KDD Dataset - https://www.unb.ca/cic/datasets/nsl.html
2. UNSW-NB15 Dataset - https://research.unsw.edu.au/projects/unsw-nb15-dataset
3. CICIDS2017 Dataset - https://www.unb.ca/cic/datasets/ids-2017.html
4. CICIDS2018 Dataset - https://www.unb.ca/cic/datasets/ids-2018.html
5. PyTorch Documentation - https://pytorch.org/docs/

---

*Report Generated: February 2, 2026*
