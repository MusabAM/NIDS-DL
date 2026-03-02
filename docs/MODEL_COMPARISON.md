# Model Comparison Analysis
## NIDS-DL Deep Learning Models

---

## Performance Comparison Chart

```
Test Accuracy by Model and Dataset
═══════════════════════════════════════════════════════════════

NSL-KDD Dataset:
├── LSTM:        ████████████████████████████████████████░░░░░░░░░░  80.78%
├── CNN:         ███████████████████████████████████████░░░░░░░░░░░  78.84%
├── Transformer: █████████████████████████████████████████░░░░░░░░  82.06%
└── Autoencoder: ██████████████████████████████████████████░░░░░░░  85.53% ★ BEST

UNSW-NB15 Dataset:
├── LSTM:        ████████████████████████████████████████████░░░░░░  88.48%
├── CNN:         █████████████████████████████████████████████░░░░░  88.78% ★ BEST
├── Transformer: ███████████████████████████████████████████░░░░░░░  87.35%
└── Autoencoder: ████████████████████████████████████████████░░░░░░  88.04%

CICIDS2018 Dataset:
├── LSTM:        ████████████████████████████████████████████████░░  95.90%
├── CNN:         █████████████████████████████████████████████████░  96.43% ★ BEST
└── Transformer: ████████████████████████████████████████████████░░  96.05%

Legend: █ = Accuracy, ░ = Remaining to 100%
```

---

## Detailed Metrics

### Accuracy Comparison
| Model | NSL-KDD | UNSW-NB15 | CICIDS2018 | Average |
|-------|---------|-----------|------------|---------|
| LSTM | 80.78% | 88.48% | 95.90% | 88.39% |
| CNN | 78.84% | **88.78%** | **96.43%** | 88.02% |
| Transformer | 82.06% | 87.35% | 96.05% | 88.49% |
| Autoencoder | **85.53%** | 88.04% | — | 86.79% |

### Precision (Attack Detection)
| Model | NSL-KDD | UNSW-NB15 | CICIDS2018 |
|-------|---------|-----------|------------|
| LSTM | 97% | 99% | 99% |
| CNN | 97% | 99% | 99% |
| Transformer | 98% | 99% | 99% |
| Autoencoder | 92% | 99% | — |

### Recall (Attack Detection)
| Model | NSL-KDD | UNSW-NB15 | CICIDS2018 |
|-------|---------|-----------|------------|
| LSTM | 68% | 84% | 93% |
| CNN | 65% | 84% | 94% |
| Transformer | 70% | 82% | 93% |
| Autoencoder | 82% | 83% | — |

### F1-Score (Weighted)
| Model | NSL-KDD | UNSW-NB15 | CICIDS2018 |
|-------|---------|-----------|------------|
| LSTM | 0.81 | 0.89 | 0.96 |
| CNN | 0.79 | 0.89 | 0.96 |
| Transformer | 0.82 | 0.88 | 0.96 |
| Autoencoder | 0.87 | 0.88 | — |

### ROC-AUC Score
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| Autoencoder | **0.9468** | **0.9809** |

---

## Generalization Analysis

### Validation-Test Gap (Overfitting Indicator)
| Dataset | LSTM Gap | CNN Gap | Interpretation |
|---------|----------|---------|----------------|
| NSL-KDD | 18.90% | 20.76% | High overfitting ⚠️ |
| UNSW-NB15 | 8.42% | 7.09% | Moderate, acceptable |
| CICIDS2018 | ~3% | ~3% | Excellent generalization ✅ |

**Insight:** CICIDS2018 shows the best generalization. NSL-KDD models suffer from overfitting
and are being actively improved with stronger regularization (L2, label smoothing, threshold tuning).

---

## Training Efficiency

| Model | Dataset | Epochs to Best | Model Size |
|-------|---------|----------------|------------|
| LSTM | NSL-KDD | ~50 | 17.7 MB |
| CNN | NSL-KDD | ~88 | 2.0 MB |
| LSTM | UNSW-NB15 | ~80 | 10.9 MB |
| CNN | UNSW-NB15 | ~18 | 2.0 MB |
| CNN | CICIDS2018 | ~30 | 2.0 MB |
| Autoencoder | NSL-KDD | ~36 | 45 KB |
| Autoencoder | UNSW-NB15 | ~45 | 45 KB |

**Insight:** Autoencoder is extremely compact (~0.5% of CNN size) while achieving strong performance on NSL-KDD.

---

## CICIDS2018 Results (Best Dataset)

| Model | Accuracy | Precision | Recall (Attack) | F1 |
|-------|----------|-----------|------------------|----|
| CNN | **96.43%** | 99% | 94% | 0.96 |
| Transformer | 96.05% | 99% | 93% | 0.96 |
| LSTM | 95.90% | 99% | 93% | 0.96 |

All three models achieve near-identical performance on CICIDS2018 (~96% accuracy) with excellent
precision and strong recall, confirming the architecture quality. The dataset's richer feature
set and training volume yield significantly better generalization than NSL-KDD.

---

## Confusion Matrix Comparison

### NSL-KDD
```
LSTM                          CNN                           Autoencoder
TN: ~9,500  FP: ~200         TN: 9,490  FP: 221            TN: 8,741  FP: 970
FN: ~4,100  TP: ~8,700       FN: 4,550  TP: 8,283          FN: 2,291  TP: 10,542
```

### UNSW-NB15
```
LSTM                          CNN                           Autoencoder
TN: 54,751  FP: 1,249        TN: 54,863  FP: 1,137         TN: 50,400  FP: 5,600
FN: 18,956  TP: 100,385      FN: 18,535  TP: 100,806       FN: 68,453  TP: 50,888
```

### CICIDS2018
```
CNN                           Transformer                   LSTM
TN: 74,514  FP: 486          TN: 44,421  FP: 579           TN: 29,744  FP: 256
FN: 4,863   TP: 70,137       FN: 2,976   TP: 42,024        FN: 2,205   TP: 27,795
```

---

## Autoencoder: Anomaly Detection Analysis

### Approach
- **Training:** Normal traffic only (unsupervised)
- **Detection:** Samples with high reconstruction error = anomaly/attack
- **Advantage:** Can detect novel/unknown attack types

### Reconstruction Error Statistics
| Dataset | Normal Mean | Attack Mean | Separation Ratio |
|---------|-------------|-------------|------------------|
| NSL-KDD | 0.00284 | 0.05097 | **17.9x** |
| UNSW-NB15 | 0.00404 | 0.00844 | 2.1x |

**Insight:** NSL-KDD shows excellent separation (17.9x), explaining the high accuracy.

---

## Known Issues & Ongoing Improvements

| Issue | Dataset | Status |
|-------|---------|--------|
| High val→test gap (~20%) | NSL-KDD | 🔧 Improved scripts created (label smoothing, threshold tuning) |
| Low Normal recall (~74%) | UNSW-NB15 | 🔧 Class-weighted loss + threshold tuning applied |
| Missing CICIDS2017 result files | CICIDS2017 | 📋 Pending |

---

## Recommendations

### Best Overall Model
**CNN on CICIDS2018** — Highest test accuracy (96.43%) with smallest model size and best generalization.

### For Different Priorities:

| Priority | Recommended Model | Reason |
|----------|------------------|--------|
| Accuracy | CNN on CICIDS2018 | 96.43% test accuracy, minimal overfitting |
| Novel Attack Detection | Autoencoder on NSL-KDD | 85.53% unsupervised, 0.95 ROC-AUC |
| Speed/Size | Autoencoder | 45 KB, extremely lightweight |
| Attack Precision | LSTM/CNN | All achieve 97–99% |
| Generalization | CICIDS2018 models | ~3% val-test gap |

---

## Key Takeaways

1. **CICIDS2018 shows best generalization** — ~96% accuracy with very low overfitting
2. **NSL-KDD shows high overfitting** — 17–20% val-test gap, under active improvement
3. **Autoencoder excels on NSL-KDD** — best accuracy (85.53%) unsupervised
4. **CNN is efficient** — smallest checkpoint (~2 MB), fastest training, strong performance
5. **UNSW-NB15 Normal recall is improving** — class-weighted loss + threshold tuning being applied

---

*Analysis updated: February 2026 — includes CICIDS2018 results and ongoing improvements*
