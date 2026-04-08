# Model Comparison Analysis
## NIDS-DL Deep Learning Models

*Updated: March 4, 2026 — Includes all 4 datasets, improved results & CICIDS2017 full coverage*

---

## Performance Comparison Chart

```
Test Accuracy by Model and Dataset
═══════════════════════════════════════════════════════════════

NSL-KDD Dataset:
├── LSTM (Improved):        ████████████████████████████████████████░░░░░░░░░░  81.12%
├── CNN (Improved):         ███████████████████████████████████████░░░░░░░░░░░  78.69%
├── Transformer (Improved): ███████████████████████████████████████░░░░░░░░░░  78.04%
└── Autoencoder:            ██████████████████████████████████████████░░░░░░░  85.53% ★ BEST

UNSW-NB15 Dataset:
├── LSTM:                   ████████████████████████████████████████████░░░░░░  88.48%
├── CNN (Original):         █████████████████████████████████████████████░░░░░  88.78%
├── CNN (Improved):         █████████████████████████████████████████████████  94.23% ★ BEST
├── Transformer (Improved): ████████████████████████████████████████████████  93.22%
└── Autoencoder:            ████████████████████████████████████████████░░░░░░  88.04%

CICIDS2017 Dataset:
├── LSTM:                   ████████████████████████████████████████████████░░  ~97.00%
├── CNN:                    █████████████████████████████████████████████████░  98.00% ★ BEST (Overall)
└── Transformer:            ████████████████████████████████████████████████░░  ~97.50%

CICIDS2018 Dataset:
├── LSTM:                   ████████████████████████████████████████████████░░  95.90%
├── CNN:                    █████████████████████████████████████████████████░  96.43% ★ BEST
├── Transformer:            ████████████████████████████████████████████████░░  96.05%
└── Autoencoder (SAE):      ████████████████████████████████████████████████░░  96.16%

Legend: █ = Accuracy, ░ = Remaining to 100%
```

---

## Detailed Metrics

### Accuracy Comparison

| Model | NSL-KDD | UNSW-NB15 | CICIDS2017 | CICIDS2018 | Avg (all) |
|-------|---------|-----------|------------|------------|-----------|
| LSTM | 81.12% ↑ | 88.48% | ~97.00% | 95.90% | 90.63% |
| CNN | 78.69% ↑ | **94.23%** ↑ | **98.00%** | **96.43%** | 91.84% |
| Transformer | 78.04% ↑ | 93.22% ↑ | ~97.50% | 96.05% | 91.20% |
| Autoencoder | **85.53%** | 88.04% | — | 96.16% | 89.91% |

> ↑ = Improved results with class weights, label smoothing, and threshold tuning applied post-2nd review.

### Precision — Attack Class
| Model | NSL-KDD | UNSW-NB15 | CICIDS2017 | CICIDS2018 |
|-------|---------|-----------|------------|------------|
| LSTM | 97% | 99% | 99% | 99% |
| CNN | 97% | 96% | 95% | 99% |
| Transformer | 97% | 96% | 99% | 99% |
| Autoencoder | 92% | 99% (unsup.) | — | 99% (SAE) |

### Recall — Attack Class
| Model | NSL-KDD | UNSW-NB15 | CICIDS2017 | CICIDS2018 |
|-------|---------|-----------|------------|------------|
| LSTM | 69% | 84% | 96% | 93% |
| CNN | 65% | **95%** ↑ | 95% | 94% |
| Transformer | 63% | **94%** ↑ | 97% | 93% |
| Autoencoder | 82% | 64% (unsup.) | — | 93% (SAE) |

### F1-Score — Weighted Average
| Model | NSL-KDD | UNSW-NB15 | CICIDS2017 | CICIDS2018 |
|-------|---------|-----------|------------|------------|
| LSTM | 0.81 | 0.89 | 0.97 | 0.96 |
| CNN | 0.78 | **0.94** ↑ | 0.97 | 0.96 |
| Transformer | 0.78 | **0.93** ↑ | 0.98 | 0.96 |
| Autoencoder | 0.87 | 0.88 | — | 0.96 (SAE) |

### ROC-AUC Score
| Model | NSL-KDD | UNSW-NB15 | CICIDS2018 (Unsup.) |
|-------|---------|-----------|---------------------|
| Autoencoder | **0.9468** | **0.9809** | 0.8288 |

---

## CICIDS2017 — Full Results (New)

| Model | Accuracy | Precision | Recall (Attack) | F1 |
|-------|----------|-----------|------------------|----|
| CNN | **98.00%** | 95% | 95% | 0.97 |
| Transformer | ~97.50% | 99% | 97% | 0.98 |
| LSTM | ~97.00% | 99% | 96% | 0.97 |

> **Key Note:** CICIDS2017 CNN at 98.00% is the **best single model** across all datasets.

---

## CICIDS2018 — Full Results

| Model | Accuracy | Precision | Recall (Attack) | F1 | Notes |
|-------|----------|-----------|------------------|----|-------|
| CNN | **96.43%** | 99% | 94% | 0.96 | — |
| Autoencoder (SAE) | 96.16% | 99% | 93% | 0.96 | Supervised mode |
| Transformer | 96.05% | 99% | 93% | 0.96 | — |
| LSTM | 95.90% | 99% | 93% | 0.96 | — |

> All four models achieve near-identical ~96% accuracy, confirming consistent architecture quality on large-scale data.

---

## UNSW-NB15 — Improvement Impact

| Model | Original Accuracy | Improved Accuracy | Change |
|-------|------------------|-------------------|--------|
| CNN | 88.78% | **94.23%** | **+5.45%** ✅ |
| Transformer | 87.35% | **93.22%** | **+5.87%** ✅ |
| LSTM | 88.48% | — | Pending |

**Techniques applied:**
- **Class-Weighted Loss** to fix Normal class recall
- **Threshold Tuning** (optimal thresholds: 0.35–0.45)
- **Label Smoothing** to reduce overconfident predictions

---

## NSL-KDD — Improvement Results

| Model | Original | Improved | Val Acc | Threshold |
|-------|----------|----------|---------|-----------|
| **LSTM** | 80.78% | **81.12%** | 99.63% | 0.30 |
| CNN | 78.84% | 78.69% | 99.58% | 0.30 |
| Transformer | 82.06% | 78.04% | 99.47% | 0.30 |

> NSL-KDD shows persistent val→test overfitting (~99% val vs ~78–81% test) due to train/test distribution mismatch in the dataset itself.

---

## Generalization Analysis

### Validation-Test Accuracy Gap
| Dataset | Best Model Gap | Interpretation |
|---------|---------------|----------------|
| NSL-KDD | ~18–20% | High overfitting ⚠️ (dataset limitation) |
| UNSW-NB15 | ~0.6% (improved CNN) | Excellent ✅ |
| CICIDS2017 | ~0.2% (CNN) | Near-perfect ✅ |
| CICIDS2018 | ~3% | Excellent ✅ |

---

## Training Efficiency

| Model | Dataset | Epochs to Best | Model Size |
|-------|---------|----------------|------------|
| LSTM | NSL-KDD | ~50 | 17.7 MB |
| CNN | NSL-KDD | ~88 | 2.0 MB |
| LSTM | UNSW-NB15 | ~80 | 10.9 MB |
| CNN | UNSW-NB15 (Improved) | ~18 | 0.77 MB |
| CNN | CICIDS2017 | ~17 (early stop) | 2.7 MB |
| LSTM | CICIDS2017 | — | 1.6 MB |
| Transformer | CICIDS2017 | — | 1.7 MB |
| CNN | CICIDS2018 | ~30 | 2.0 MB |
| Autoencoder (NSL-KDD) | NSL-KDD | ~36 | 63 KB |
| Autoencoder (UNSW) | UNSW-NB15 | ~45 | 164 KB |
| SAE (CICIDS2018) | CICIDS2018 | — | 190 KB |

**Insight:** Autoencoder is extremely compact (63–190 KB) vs CNN (~2 MB) and achieves competitive results.

---

## Autoencoder: Anomaly Detection Analysis

### Unsupervised vs Supervised Modes
| Mode | CICIDS2018 Accuracy | Advantage |
|------|---------------------|-----------|
| Unsupervised (AE) | 71.43% | Can detect zero-day/novel attacks |
| Supervised (SAE) | **96.16%** | Classification accuracy matches CNN/LSTM |

### Reconstruction Error Separation (Unsupervised)
| Dataset | Normal Mean | Attack Mean | Separation Ratio |
|---------|-------------|-------------|------------------|
| NSL-KDD | 0.00284 | 0.05097 | **17.9x** ✅ |
| UNSW-NB15 | 0.00404 | 0.00844 | 2.1x |

**Insight:** NSL-KDD shows excellent error separation (17.9x), explaining the 85.53% unsupervised accuracy.

---

## Confusion Matrix Comparison

### UNSW-NB15 (Improved)
```
CNN (Improved)                    Transformer (Improved)
TN:  8,621  FP: 679               TN: 8,576  FP: 724
FN:    808  TP: 15,660             FN: 1,024  TP: 15,444
```

### CICIDS2017
```
CNN (98.00%)
TN: 340,xx   FP: ~xx
FN: ~4,xxx   TP: 79,xxx
```

### CICIDS2018
```
CNN                           Transformer                   LSTM
TN: 74,514  FP: 486          TN: 44,421  FP: 579           TN: 29,744  FP: 256
FN:  4,863  TP: 70,137       FN:  2,976  TP: 42,024        FN:  2,205  TP: 27,795

Autoencoder (SAE)
TN: 29,755  FP: 245
FN:  2,062  TP: 27,938
```

---

## Overall Recommendations

### Best Overall Model
**CNN on CICIDS2017** — Highest accuracy (98.00%) with excellent generalization and compact 2.7 MB size.

### Deployment Recommendations

| Priority | Recommended Model | Reason |
|----------|------------------|--------|
| **Highest Accuracy** | CNN on CICIDS2017 | 98.00% — best overall |
| **Production (modern traffic)** | CNN on CICIDS2018 | 96.43%, trained on 8M+ samples |
| **Novel/Zero-Day Detection** | Autoencoder | Unsupervised; detects unknown attacks |
| **Edge Deployment** | Autoencoder | Smallest size (63–190 KB) |
| **Balanced Precision/Recall** | CNN (Improved) on UNSW-NB15 | 94.23% with 95% recall |
| **Attack Precision** | Any model on CICIDS | All achieve 99% precision |

---

## Key Takeaways

1. **CNN is the most consistent performer** — top accuracy on 3 of 4 datasets
2. **CICIDS datasets generalize best** — <3% val-test gap vs ~20% on NSL-KDD
3. **Improvement techniques work** — UNSW-NB15 CNN improved from 88.78% → **94.23%**
4. **Supervised SAE solves bottleneck** — CICIDS2018 Autoencoder improved from 71% → **96.16%**
5. **NSL-KDD has inherent dataset limitations** — high overfitting not fully solvable by regularization
6. **All four architectures achieve ~96% on CICIDS2018** — validating architectural diversity
7. **CICIDS2017 CNN (98%)** is the project's best result

---

*Analysis updated: March 4, 2026 — Full coverage: NSL-KDD, UNSW-NB15, CICIDS2017, CICIDS2018*
