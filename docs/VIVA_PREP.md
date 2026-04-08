# NIDS-DL Viva Preparation Guide
## Network Intrusion Detection using Deep Learning

**Prepared for: 3rd Review Viva | Date: March 5, 2026**

---

# PART 1: PROJECT EXPLAINED SIMPLY

## What is this project about? (In plain English)

Imagine millions of people using the internet every second. Among them, some are **hackers** trying to break into computer systems. A **Network Intrusion Detection System (NIDS)** watches all the network traffic and raises an alarm when it sees something suspicious — like a security camera for your network.

Traditional systems use a **rulebook**: "if traffic looks like X, it's an attack." Our project uses **Deep Learning** — the computer *learns by itself* what normal and attack traffic looks like, without needing a pre-written rulebook. This makes it smarter and capable of detecting **new kinds of attacks** it has never seen before.

---

## The Big Picture

```
Internet Traffic
      |
      v
[Feature Extraction] — 77 properties of each network connection
(How long? How many bytes? What protocol? etc.)
      |
      v
[Deep Learning Model] — Learned from millions of examples
(CNN / LSTM / Transformer / Autoencoder)
      |
      v
[Decision: NORMAL or ATTACK]
      |
      v
[Dashboard Alert]
```

---

## The 4 Datasets (Our Testing Grounds)

Think of datasets as **exam papers of different difficulty**:

| Dataset | Size | Think of it as... | Our Result |
|---------|------|-------------------|------------|
| NSL-KDD | ~148K samples | Old, simple exam | 85.53% (AE) |
| UNSW-NB15 | ~257K samples | Moderate difficulty | **94.23%** (CNN) |
| CICIDS2017 | 2.8 Million | Modern, realistic | **98.00%** (CNN) |
| CICIDS2018 | 8.2 Million | Large-scale, hardest | **96.43%** (CNN) |

---

## The 4 Models (Our Security Guards)

### 1. CNN — The Pattern Spotter
- Looks at **spatial patterns** in network features (like finding fingerprints)
- Very fast, very accurate (achieves 98% on CICIDS2017)
- Best overall performer in this project

### 2. LSTM — The Memory Expert
- Looks at **sequences over time** — notices if a pattern repeats or builds up
- Like a guard who remembers the last 100 people who entered and notices suspicious trends
- Great at detecting slow, persistent attacks

### 3. Transformer — The Attention Master
- Figures out **which features matter most** for each specific sample
- Like a detective who focuses on the most relevant clues
- Strong on CICIDS2017 (~97.5%), state-of-the-art architecture (same as GPT/BERT)

### 4. Autoencoder — The Anomaly Hunter
- Trained **only on normal traffic** (unsupervised)
- Learns to "recreate" normal traffic patterns
- When it can't recreate a sample well → it's probably an attack
- Can detect **unknown, zero-day attacks** that the other models would miss

---

## Key Results (What We Achieved)

| Achievement | Value |
|-------------|-------|
| Best Accuracy | **98.00%** (CNN on CICIDS2017) |
| Best Large-Scale | **96.43%** (CNN on CICIDS2018 — 8.2M samples) |
| Best Improvement | UNSW-NB15 CNN: 88.78% → **94.23%** (+5.45%) |
| Attack Precision | **99%** (all CICIDS models) |
| Quantum ML | VQC implemented using PennyLane (exploratory) |

---

# PART 2: TECHNICAL DEEP DIVE

## How Each Model Actually Works

### CNN Architecture (1D Convolution)
```
Input: [batch_size, 1, 77_features]
   → Conv1D(1→64, kernel=3) + BatchNorm + ReLU
   → Conv1D(64→128, kernel=3) + BatchNorm + ReLU
   → Conv1D(128→256, kernel=3) + BatchNorm + ReLU
   → AdaptiveAvgPool1d
   → FC(2048→256) + Dropout(0.3)
   → FC(256→64) + Dropout(0.3)
   → FC(64→2) → Softmax
Output: [Normal, Attack] probabilities
```

### LSTM Architecture (Bidirectional)
```
Input: [batch, sequence, 77_features]
   → 3x Bidirectional LSTM (256 hidden) with attention
   → Dropout(0.4) + BatchNorm
   → FC(512→2)
Output: [Normal, Attack] probabilities
```

### Transformer Architecture
```
Input: [batch, 77_features]
   → Linear Embedding (77→64)
   → 3x TransformerEncoder blocks:
      - Multi-Head Self-Attention (4 heads)
      - Pre-LayerNorm (for stability)
      - Feed-Forward (64→128→64)
      - Dropout(0.3)
   → Global Average Pooling
   → FC(64→2)
Output: [Normal, Attack] probabilities
```

### Autoencoder Architecture
```
ENCODER: input → Dense(64) → Dense(32) → Dense(16) → Dense(8) [latent]
DECODER: Dense(8) → Dense(16) → Dense(32) → Dense(64) → input
Detection: if reconstruction_error > threshold → ATTACK
```

---

## Key Techniques We Used

### 1. Focal Loss
- **Why:** CICIDS2017 has 80% normal, 20% attack → model ignores attack
- **What it does:** Focuses training on hard-to-classify samples
- **Formula concept:** Loss = −(1−pt)^γ × log(pt), α=0.25, γ=2

### 2. Class-Weighted Loss
- **Why:** UNSW-NB15 had very low recall for Normal class
- **What it does:** Penalizes errors on minority class more heavily
- **Result:** CNN improved 88.78% → 94.23%

### 3. Threshold Tuning
- **Why:** Default 0.5 threshold doesn't give best precision/recall balance
- **What it does:** Finds optimal threshold via ROC curve analysis
- **Used:** 0.30 for NSL-KDD, 0.35–0.45 for UNSW-NB15

### 4. Label Smoothing
- **Why:** Prevents overconfident predictions and reduces overfitting
- **What it does:** Instead of hard 0/1 labels, uses 0.1/0.9
- **Effect:** NSL-KDD val-test gap reduced slightly

### 5. Git LFS (Large File Storage)
- **Why:** Model files are 1–17 MB, too large for normal git
- **What it does:** Stores files by reference on GitHub
- **Used for:** All .pth/.pt model weight files

---

## Quantum ML Section (VQC)

### What is a Variational Quantum Classifier?
```
Classical Data (122 features)
      ↓
PCA reduction (122 → 8 features)
      ↓
Quantum Circuit (8 qubits):
  - Angle Encoding: RY(θ_i) on each qubit
  - 4x Strongly Entangling Layers (trainable params)
  - Pauli-Z measurements
      ↓
Classical Postprocessing:
  Dense(8→16) → ReLU → Dense(16→8) → Dense(8→2)
      ↓
Output: Normal / Attack
```

### Why Quantum?
- Quantum circuits can potentially explore **exponentially large feature spaces**
- Entanglement creates **correlations** between features that classical models miss
- **Trade-off:** Much slower (~30-60 min for 10K samples vs seconds classically)

---

# PART 3: VIVA QUESTIONS & ANSWERS

## Basic / Conceptual Questions

**Q1: What is a Network Intrusion Detection System (NIDS)?**
> A NIDS monitors network traffic in real-time and identifies suspicious patterns that indicate cyberattacks. It can be signature-based (rule-based) or anomaly-based (learning-based). Our project uses deep learning-based anomaly detection.

**Q2: Why did you use Deep Learning instead of traditional ML?**
> Traditional ML (like SVM, Random Forest) requires manual feature engineering and struggles with high-dimensional, imbalanced data at scale. Deep learning:
> - Automatically learns hierarchical features
> - Scales to millions of samples (CICIDS2018: 8.2M)
> - Generalizes better to new attack variants
> - Achieves higher accuracy (98% vs ~90% for classical methods)

**Q3: What is binary classification in your context?**
> We classify each network flow as either **Normal** (0) or **Attack** (1). This is binary because there are only two classes.

**Q4: What is the difference between precision and recall?**
> - **Precision:** Of all flows flagged as attacks, how many were actually attacks? (Avoid false alarms)
> - **Recall:** Of all actual attacks, how many did we catch? (Don't miss threats)
> - **F1-Score:** Harmonic mean of both — balances the two

**Q5: Why is 98% accuracy impressive for network intrusion detection?**
> With 2.8 million samples and 80:20 class imbalance, achieving 98% means:
> - 99% recall on benign (very few false alarms)
> - 95% recall on attacks (most threats caught)
> - High generalization: val and test accuracy differ by < 0.2%
> Older classical methods rarely exceed 90% on similar datasets.

---

## Dataset Questions

**Q6: Why did you use 4 different datasets?**
> Each dataset tests different aspects:
> - NSL-KDD: Classic benchmark, simple (easy baseline)
> - UNSW-NB15: Modern, 9 attack types, realistic traffic
> - CICIDS2017: Very large (2.8M), modern attack types including web attacks
> - CICIDS2018: Largest (8.2M), most recent, best real-world simulation

**Q7: What does "overfitting" mean and why does NSL-KDD have it?**
> Overfitting: Model performs well on training data but poorly on test data.
> NSL-KDD has ~99% val accuracy but only ~81% test accuracy because:
> - The train and test sets have different statistical distributions
> - The dataset is older and simpler — models memorize rather than generalize
> - This is a known limitation of NSL-KDD in the research community

**Q8: What kind of attacks are in CICIDS2017?**
> - DDoS (Distributed Denial of Service)
> - Port Scanning
> - Web Attacks: XSS, SQL Injection, Brute Force
> - Infiltration attacks
> - Bot/botnet traffic
> - FTP-Patator and SSH-Patator (credential brute force)

**Q9: What preprocessing did you do on the datasets?**
> 1. Removed NaN/Infinity values
> 2. Dropped non-numeric columns (Flow ID, IPs, timestamps)
> 3. StandardScaler normalization (mean=0, std=1)
> 4. Saved scaler as `.pkl` file for inference consistency
> 5. Stratified train/val/test split (70/15/15 or 70/15/15)

---

## Model Architecture Questions

**Q10: Why did you use 1D CNN instead of 2D CNN?**
> Network traffic data is a 1D feature vector (77 or 76 features per flow), not a 2D image. 1D convolutions slide across the feature dimension to detect local patterns within the feature space, which is more appropriate.

**Q11: What is the role of BatchNorm in your models?**
> Batch Normalization normalizes activations within each mini-batch, which:
> - Stabilizes training and allows higher learning rates
> - Reduces internal covariate shift
> - Acts as a mild regularizer

**Q12: Why Bidirectional LSTM?**
> A standard LSTM reads a sequence left-to-right. A Bidirectional LSTM reads both left-to-right AND right-to-left, then combines both — capturing context from both directions. For network features (not true time series), this helps extract richer feature interactions.

**Q13: What is self-attention in the Transformer?**
> Self-attention lets the model weigh the importance of each feature relative to every other feature in the same input. Mathematically:
> Attention(Q, K, V) = softmax(QK^T / √d_k) × V
> This allows the model to focus on the most discriminative features for each specific input.

**Q14: What is the difference between supervised and unsupervised Autoencoder?**
> - **Unsupervised:** Train only on normal traffic. Detection by reconstruction error (high error = attack). Can detect zero-day attacks but lower accuracy (~71%).
> - **Supervised (SAE):** Add a classification head + use both normal and attack labels. Gets the anomaly detection benefit AND labeled training signal. Achieved 96.16% accuracy.

---

## Technical Implementation Questions

**Q15: What is Focal Loss and why did you use it?**
> Focal Loss = −α(1−pt)^γ × log(pt)
> - Standard cross-entropy treats all samples equally
> - With 80:20 imbalance, model tends to predict "Normal" always and gets 80% accuracy for free
> - Focal Loss reduces the weight of easy-to-classify samples and focuses on hard ones
> - α=0.25, γ=2 are hyperparameters that control this behavior

**Q16: What is the ROC-AUC score?**
> - ROC = Receiver Operating Characteristic curve (TPR vs FPR at different thresholds)
> - AUC = Area Under the Curve
> - AUC=1.0 is perfect, AUC=0.5 is random guessing
> - Our Autoencoder achieved AUC=0.9468 on NSL-KDD and 0.9809 on UNSW-NB15

**Q17: What is Git LFS and why did you need it?**
> Git Large File Storage is an extension that replaces large files in your repo with text pointers and stores the actual file content on a remote server. We needed it because model `.pth` files range from 63 KB to 17 MB — too large for standard git and especially for GitHub's 100MB file limit.

**Q18: What is early stopping?**
> A regularization technique: we monitor validation loss after each epoch. If val loss doesn't improve for N epochs (patience), training stops. This prevents overfitting by halting before the model memorizes training data.

**Q19: What is AdamW optimizer?**
> Adam optimizer with decoupled Weight Decay (L2 regularization). Standard Adam applies weight decay incorrectly (adds it to the gradient). AdamW fixes this, leading to better generalization especially on complex architectures.

**Q20: What is threshold tuning?**
> By default, we classify as "Attack" if the model outputs probability > 0.5. But this isn't always optimal. We:
> 1. Generate predictions on validation set
> 2. Try thresholds from 0.1 to 0.9
> 3. Pick the threshold that maximizes F1-score
> For NSL-KDD, optimal threshold was 0.30 (more aggressive attack detection)

---

## Quantum ML Questions

**Q21: Why did you explore Quantum Machine Learning?**
> Quantum computers can represent exponentially large state spaces using superposition and entanglement. A Variational Quantum Classifier (VQC) can potentially:
> - Explore feature correlations classical computers can't efficiently compute
> - Use quantum kernels that map data to very high-dimensional Hilbert spaces
> This is exploratory research — we wanted to compare QML vs classical on the same task.

**Q22: What is a qubit and why 8 qubits?**
> A qubit is the quantum equivalent of a bit — but instead of 0 or 1, it can be in a **superposition** of both simultaneously. 8 qubits can represent 2^8 = 256 states simultaneously.
> We used 8 qubits because our PCA reduced features to 8 — each feature encodes to one qubit via angle encoding.

**Q23: What is PCA and why was it needed for QML?**
> Principal Component Analysis: reduces dimensionality by finding directions of maximum variance. We went from 122 features → 8 features because:
> - Current quantum simulators are only feasible with few qubits
> - Each qubit encodes one feature (angle encoding)
> - More qubits = exponentially longer simulation time

---

## Application / Dashboard Questions

**Q24: What does your Streamlit dashboard do?**
> The Pro Dashboard provides:
> 1. **Live Threat Analysis:** Real-time packet capture → feature extraction → model prediction → alert
> 2. **Batch Processing:** Upload CSV/PCAP → process all flows → generate threat report
> 3. **Model Selection:** Choose between CNN, LSTM, Transformer, or Autoencoder
> 4. **Visualizations:** Attack distribution charts, threat level indicators, confusion matrices

**Q25: How does the live sniffer work?**
> The `live_sniffer.py` script:
> 1. Captures raw network packets using socket/scapy
> 2. Extracts 77 CICIDS-compatible features (duration, byte counts, protocol, flags, etc.)
> 3. Scales features using the pre-saved scaler (`cicids2017_scaler.pkl`)
> 4. Passes to loaded CNN model for real-time classification
> 5. Displays result: Normal or Attack (with confidence %)

**Q26: Why FastAPI for the backend?**
> FastAPI is a high-performance Python web framework that:
> - Generates automatic API documentation (Swagger UI)
> - Handles async requests efficiently for real-time inference
> - Has built-in data validation with Pydantic
> - Integrates cleanly with PyTorch model inference

---

## Comparison & Analysis Questions

**Q27: Which model would you recommend for real-world deployment and why?**
> **CNN on CICIDS2018** because:
> - 96.43% accuracy on modern, large-scale data (8.2M samples)
> - Smallest model size (~2 MB) for fast loading
> - Fastest inference time
> - Trained on the most realistic dataset
> - 99% precision means very few false alarms in production

**Q28: Why does the Autoencoder perform worse on CICIDS2018 in unsupervised mode?**
> CICIDS2018 has 26.6% attack traffic vs NSL-KDD's different distribution. When we train the AE on only normal traffic and test on both:
> - The reconstruction error gap between normal and attack is smaller (2.1x vs 17.9x on NSL-KDD)
> - This makes threshold-based detection unreliable → 71% accuracy
> Solution: Switch to **Supervised Autoencoder (SAE)** — adds classification head, trains on labeled data → 96.16%

**Q29: What is the biggest challenge in this project?**
> **Class imbalance** across all datasets — benign traffic always dominates (73-80%). Solutions used:
> - Focal Loss for CICIDS datasets
> - Class-weighted cross-entropy for NSL-KDD/UNSW
> - Threshold tuning for optimal operating point
> - Stratified sampling during train/val/test splits

**Q30: What would you improve if given more time?**
> 1. **Ensemble:** Combine CNN + LSTM + Autoencoder outputs via weighted voting
> 2. **Multi-class:** Detect specific attack types (DDoS vs Port Scan vs Web Attack)
> 3. **Variational Autoencoder (VAE):** Better anomaly scoring with latent distribution modeling
> 4. **Real quantum hardware:** Test VQC on IBM Quantum or Google Quantum AI
> 5. **Federated Learning:** Train across multiple network nodes without sharing raw data

---

# PART 4: QUICK REFERENCE CHEAT SHEET

## Numbers to Remember

| Fact | Value |
|------|-------|
| Best accuracy | **98.00%** — CNN, CICIDS2017 |
| Largest dataset | **8.28M samples** — CICIDS2018 |
| Most features | **77** — CICIDS2017 |
| Total models trained | **16+** (4 models × 4 datasets + improved variants) |
| Quantum qubits | **8 qubits** |
| UNSW-NB15 improvement | **+5.45%** (88.78% → 94.23%) |
| Autoencoder improvement (supervised) | **+25%** (71% → 96.16%) |
| NSL-KDD AE ROC-AUC | **0.9468** |
| UNSW-NB15 AE ROC-AUC | **0.9809** |

## Key Terms Flash Cards

| Term | Simple Meaning |
|------|---------------|
| NIDS | Security camera for your network |
| CNN | Finds patterns in features (like fingerprints) |
| LSTM | Remembers patterns over time |
| Transformer | Decides which features to focus on |
| Autoencoder | Learns normal, flags abnormal |
| Focal Loss | Focuses attention on hard examples |
| ROC-AUC | Overall detection quality (0.5=random, 1.0=perfect) |
| Precision | Of alerts raised, how many were real attacks |
| Recall | Of real attacks, how many did we catch |
| F1-Score | Balance between precision and recall |
| Overfitting | Good on training, bad on test |
| PCA | Dimension reduction for quantum |
| VQC | Quantum version of a classifier |
| Git LFS | Storage for large model files |
| Early Stopping | Stop training before overfitting |
| Threshold Tuning | Find best decision boundary |

---

*Good Luck Tomorrow! You've got this! 💪*

*Project: NIDS-DL — Deep Learning for Network Security*
*Viva Date: March 5, 2026*
