# Model Comparison Analysis
## NIDS-DL Deep Learning Models

---

## Performance Comparison Chart

```
Test Accuracy by Model and Dataset
═══════════════════════════════════════════════════════════════

NSL-KDD Dataset:
├── LSTM:  ████████████████████████████████████████░░░░░░░░░░  80.78%
└── CNN:   ███████████████████████████████████████░░░░░░░░░░░  78.84%

UNSW-NB15 Dataset:
├── LSTM:  ████████████████████████████████████████████░░░░░░  88.48%
└── CNN:   █████████████████████████████████████████████░░░░░  88.78% ★ BEST

Legend: █ = Accuracy, ░ = Remaining to 100%
```

---

## Detailed Metrics

### Accuracy Comparison
| Model | NSL-KDD | UNSW-NB15 | Average |
|-------|---------|-----------|---------|
| LSTM | 80.78% | 88.48% | 84.63% |
| CNN | 78.84% | **88.78%** | 83.81% |

### Precision (Attack Detection)
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| LSTM | 97% | 99% |
| CNN | 97% | 99% |

### Recall (Attack Detection)
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| LSTM | 68% | 84% |
| CNN | 65% | 84% |

### F1-Score (Weighted)
| Model | NSL-KDD | UNSW-NB15 |
|-------|---------|-----------|
| LSTM | 0.81 | 0.89 |
| CNN | 0.79 | 0.89 |

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

**Insight:** CNN is 5-9x more compact while achieving comparable performance.

---

## Confusion Matrix Comparison

### NSL-KDD
```
LSTM                          CNN
TN: ~9,500  FP: ~200         TN: 9,490  FP: 221
FN: ~4,100  TP: ~8,700       FN: 4,550  TP: 8,283
```

### UNSW-NB15
```
LSTM                          CNN
TN: 54,751  FP: 1,249        TN: 54,863  FP: 1,137
FN: 18,956  TP: 100,385      FN: 18,535  TP: 100,806
```

---

## Recommendations

### Best Overall Model
**CNN on UNSW-NB15** - Highest test accuracy (88.78%) with smallest model size (2 MB)

### For Different Priorities:

| Priority | Recommended Model | Reason |
|----------|------------------|--------|
| Accuracy | CNN on UNSW-NB15 | 88.78% test accuracy |
| Speed/Size | CNN | 8x smaller than LSTM |
| Attack Precision | Any | All achieve 97-99% |
| Generalization | UNSW-NB15 models | Lower overfitting |

---

## Key Takeaways

1. **Dataset matters more than architecture** - Both models improved ~10% on UNSW-NB15
2. **CNN is efficient** - Smaller, faster, comparable performance
3. **High precision achieved** - 97-99% attack detection precision across all models
4. **Recall trade-off** - ~16% of attacks missed (84% recall)
5. **Normal detection excellent** - 98% recall for normal traffic

---

*Analysis generated from experimental results*
