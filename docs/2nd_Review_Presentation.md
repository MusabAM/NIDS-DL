# NIDS-DL Project - 2nd Review (50% Implementation)
## Network Intrusion Detection using Deep Learning

**Date:** February 7, 2026  
**Progress:** 30% → 50%  
**Focus:** CIC-IDS 2017 Dataset Implementation

---

## 1. Project Overview

This project implements deep learning models for **Network Intrusion Detection Systems (NIDS)** using four benchmark datasets and four neural network architectures.

### Datasets
| Dataset | Status | Samples |
|---------|--------|---------|
| NSL-KDD | ✅ Completed (30%) | 148,517 |
| UNSW-NB15 | ✅ Completed (30%) | 257,673 |
| **CICIDS2017** | ✅ **Completed (50%)** | **2,830,743** |
| CICIDS2018 | 🔜 Upcoming | 8,284,254 |

### Model Architectures
- ✅ **CNN** (Convolutional Neural Network)
- ✅ **LSTM** (Long Short-Term Memory)
- ✅ **Transformer** (Self-Attention)
- ✅ **Autoencoder** (Anomaly Detection)

---

## 2. Progress from 30% to 50%

### What Was Completed in 30% Review
- Literature survey & research
- Problem definition
- System design
- CUDA environment setup
- NSL-KDD dataset training (all 4 models)
- UNSW-NB15 dataset training (all 4 models)

### What's New in 50% Review (CIC-IDS 2017)

#### 2.1 Dataset Acquisition & Preprocessing
- Downloaded **8 CSV files** from CICIDS2017 repository
- Combined all files into single dataset: **2,830,743 samples**
- Cleaned NaN/Inf values: Remaining **2,827,876 samples**
- Dropped administrative columns (Flow ID, IPs, Timestamp)
- Final features: **77 numerical features**

#### 2.2 Data Distribution
| Class | Samples | Percentage |
|-------|---------|------------|
| Benign | 2,271,320 | 80.3% |
| Attack | 556,556 | 19.7% |

#### 2.3 Attack Types in CICIDS2017
- DDoS
- Port Scan
- Web Attacks (XSS, SQL Injection, Brute Force)
- Infiltration
- Bot
- FTP/SSH Patator

---

## 3. Model Training on CICIDS2017

### 3.1 CNN Classifier (Best Performer)

**Architecture:**
```
Conv1D(1 → 64) + BatchNorm + ReLU
Conv1D(64 → 128) + BatchNorm + ReLU  
Conv1D(128 → 256) + BatchNorm + ReLU
AdaptiveAvgPool1D
FC(2048 → 256) + Dropout(0.3)
FC(256 → 64) + Dropout(0.3)
FC(64 → 2) - Output
```

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Batch Size | 1024 |
| Epochs | 20 (Early Stop at 17) |
| Optimizer | Adam (lr=0.001) |
| Loss | Focal Loss (α=0.25, γ=2) |
| Scheduler | ReduceLROnPlateau |
| GPU | NVIDIA GeForce RTX 4050 |
| Training Time | ~72 sec/epoch |

**Dataset Split:**
| Split | Samples |
|-------|---------|
| Train | 1,979,513 (70%) |
| Validation | 424,181 (15%) |
| Test | 424,182 (15%) |

---

## 4. Results - CICIDS2017 CNN

### 🎯 Test Accuracy: **98.00%**

### Classification Report
```
              precision    recall  f1-score   support

      Benign       0.99      0.99      0.99    340,698
      Attack       0.95      0.95      0.95     83,484

    accuracy                           0.98    424,182
   macro avg       0.97      0.97      0.97    424,182
weighted avg       0.98      0.98      0.98    424,182
```

### Key Metrics
| Metric | Benign | Attack |
|--------|--------|--------|
| Precision | 99% | 95% |
| Recall | 99% | 95% |
| F1-Score | 0.99 | 0.95 |

### Training Progression
| Epoch | Train Acc | Val Acc | Time |
|-------|-----------|---------|------|
| 1 | 96.20% | 97.12% | 73s |
| 8 | 97.43% | 97.71% | 72s |
| 12 | 97.78% | **98.20%** | 72s |

---

## 5. Comparison with Previous Datasets

| Dataset | Best Model | Test Accuracy | Improvement |
|---------|------------|---------------|-------------|
| NSL-KDD | Autoencoder | 85.53% | Baseline |
| UNSW-NB15 | CNN | 88.78% | +3.25% |
| **CICIDS2017** | **CNN** | **98.00%** | **+9.22%** |

### Key Insights
1. **CICIDS2017 shows best generalization** - minimal gap between validation and test accuracy
2. **CNN outperforms all models** on this larger, more modern dataset
3. **Focal Loss** effectively handles class imbalance (80:20 ratio)
4. **Large-scale processing** validated (2.8M+ samples)

---

## 6. Technical Implementation

### Files Created/Updated

| File | Description |
|------|-------------|
| `notebooks/01_data_exploration/03_cicids2017_exploration.ipynb` | Data analysis |
| `notebooks/02_classical_dl/03_cnn_cicids2017.ipynb` | CNN training |
| `notebooks/02_classical_dl/03_cnn_cicids2017_inference.ipynb` | Model testing |
| `results/models/best_cnn_cicids2017.pth` | Saved model weights |
| `data/raw/cicids2017/` | 8 CSV files |

### Code Highlights
- StandardScaler for feature normalization
- Stratified train-test split for class balance
- FocalLoss for handling class imbalance
- Early stopping with patience=5
- Learning rate scheduler

---

## 7. Timeline Update (Gantt Chart)

| Phase | Timeline | Status |
|-------|----------|--------|
| Literature Survey | Dec 2025 - Jan 2026 | ✅ |
| System Design | Dec 2025 - Jan 2026 | ✅ |
| NSL-KDD & UNSW-NB15 | Jan 2026 | ✅ 30% |
| **CICIDS2017** | **Jan - Feb 4, 2026** | **✅ 50%** |
| CICIDS2018 | Feb 9 - Mar 5, 2026 | 🔜 |
| Quantum Integration | Feb 9 - Mar 5, 2026 | 🔜 |
| Dashboard & Docs | Mar - Apr 2026 | 🔜 |

---

## 8. Next Steps (50% → 100%)

### Phase 3: CICIDS2018 (Feb 9 - Mar 5)
- Process 8.2M+ samples
- Train all 4 models
- Compare with CICIDS2017 results

### Phase 4: Quantum Computing (Feb 9 - Mar 5)
- Implement quantum circuits with PennyLane
- Variational quantum classifiers
- Hybrid classical-quantum models

### Phase 5: Finalization (Mar - Apr)
- Streamlit dashboard
- Real-time inference
- Performance evaluation
- Documentation & report

---

## 9. Conclusion

**50% Milestone Achieved:**
- ✅ Successfully processed 2.8M+ CICIDS2017 samples
- ✅ Achieved **98.00% accuracy** with CNN
- ✅ Best performance across all 3 datasets tested
- ✅ Validated large-scale deep learning pipeline

The project is on track for completion by April 2026.

---

*Report prepared for: 2nd Internal Review*  
*Project: NIDS-DL - Deep Learning for Network Security*  
*Date: February 7, 2026*
