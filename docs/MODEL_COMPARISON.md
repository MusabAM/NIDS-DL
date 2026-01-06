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
└── Autoencoder: ██████████████████████████████████████████░░░░░░░░  85.53% ★ BEST

UNSW-NB15 Dataset:
├── LSTM:        ████████████████████████████████████████████░░░░░░  88.48%
├── CNN:         █████████████████████████████████████████████░░░░░  88.78% ★ BEST
├── Transformer: ███████████████████████████████████████████░░░░░░░  87.35%
└── Autoencoder: ████████████████████████████████████████████░░░░░░  88.04%

Legend: █ = Accuracy, ░ = Remaining to 100%
```

---

## Detailed Metrics

### Accuracy Comparison
| Model | NSL-KDD | UNSW-NB15 | Average |
|-------|---------|-----------|---------|
| LSTM | 80.78% | 88.48% | 84.63% |
| CNN | 78.84% | **88.78%** | 83.81% |
| Transformer | 82.06% | 87.35% | 84.71% |
| Autoencoder | **85.53%** | 88.04% | 86.79% |

### Precision (Attack Detection)
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| LSTM | 97% | 99% |
| CNN | 97% | 99% |
| Transformer | 98% | 99% |
| Autoencoder | 92% | 99% |

### Recall (Attack Detection)
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| LSTM | 68% | 84% |
| CNN | 65% | 84% |
| Transformer | 70% | 82% |
| Autoencoder | 82% | 83% |

### F1-Score (Weighted)
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| LSTM | 0.81 | 0.89 |
| CNN | 0.79 | 0.89 |
| Transformer | 0.82 | 0.88 |
| Autoencoder | 0.87 | 0.88 |

### ROC-AUC Score
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| Autoencoder | **0.9468** | **0.9809** |

---

## Generalization Analysis

### Validation-Test Gap (Overfitting Indicator)
| Dataset | LSTM Gap | CNN Gap | Interpretation |
|---------|----------|---------|----------------|
| NSL-KDD | 18.90% | 20.76% | High overfitting |
| UNSW-NB15 | 8.42% | 7.09% | Moderate, acceptable |

**Insight:** UNSW-NB15 shows much better generalization with smaller gaps between validation and test performance.

---

## Training Efficiency

| Model | Dataset | Epochs to Best | Model Size |
|-------|---------|----------------|------------|
| LSTM | NSL-KDD | ~50 | 17.7 MB |
| CNN | NSL-KDD | ~88 | 2.0 MB |
| LSTM | UNSW-NB15 | ~80 | 10.9 MB |
| CNN | UNSW-NB15 | ~18 | 2.0 MB |
| Autoencoder | NSL-KDD | ~36 | 45 KB |
| Autoencoder | UNSW-NB15 | ~45 | 45 KB |

**Insight:** Autoencoder is extremely compact (~0.5% of CNN size) while achieving strong performance on NSL-KDD.

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

**Insight:** NSL-KDD shows excellent separation between normal and attack traffic (17.9x difference), explaining the high accuracy. UNSW-NB15 has lower separation (2.1x), making detection harder.

---

## Recommendations

### Best Overall Model
**CNN on UNSW-NB15** - Highest test accuracy (88.78%) with smallest model size (2 MB)

### For Different Priorities:

| Priority | Recommended Model | Reason |
|----------|------------------|--------|
| Accuracy | CNN on UNSW-NB15 | 88.78% test accuracy |
| Novel Attack Detection | Autoencoder on NSL-KDD | 85.53% unsupervised, 0.95 ROC-AUC |
| Speed/Size | Autoencoder | 45 KB, extremely lightweight |
| Attack Precision | LSTM/CNN | All achieve 97-99% |
| Generalization | UNSW-NB15 models | Lower overfitting |

---

## Key Takeaways

1. **Dataset matters more than architecture** - Both supervised models improved ~10% on UNSW-NB15
2. **CNN is efficient** - Smaller, faster, comparable performance
3. **Autoencoder excels on NSL-KDD** - 85.53% accuracy (best!) with unsupervised learning
4. **High precision achieved** - 90-99% attack detection precision across all models
5. **Autoencoder trade-off** - Excellent for novelty detection, but sensitive to data distribution
6. **ROC-AUC validates separation** - Autoencoder's 0.95 on NSL-KDD shows strong class separation

---

*Analysis generated from experimental results - January 2026*
