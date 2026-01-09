# Literature Survey: Deep Learning for Network Intrusion Detection

## 1. Introduction

The field of Network Intrusion Detection Systems (NIDS) has evolved significantly with the advent of Deep Learning (DL). Traditional systems based on statistical methods or simple machine learning often struggle with the complexity and volume of modern network traffic, particularly when facing zero-day exploits.

**NIDS-DL**, the project under development, aims to bridge critical gaps identified in current literature. While many existing solutions achieve high accuracy on older datasets like NSL-KDD or UNSW-NB15, they often lack the robustness required for near-real-time detection of novel attacks in modern environments.

This survey reviews similar projects and research papers, comparing their methodologies with the goals of NIDS-DL:
1.  **High Accuracy with Low False Positives.**
2.  **Zero-Day Exploit Detection** in Near-Real-Time.
3.  **Advanced Ensembling** (combining various DL models and exploring Quantum integration).
4.  **Modern Dataset Focus** (primary design on **CICIDS2018**, moving beyond legacy datasets).

## 2. Comparative Analysis

The following table compares NIDS-DL with prevalent approaches found in recent literature.

| Title / Project Theme                                 | Author / Source (Representative) | Problems Addressed                                                                         | Implementation                                                                                                                                | Limitations                                                                                                                           |
| :---------------------------------------------------- | :------------------------------- | :----------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Hybrid Deep Learning Investigation for NIDS**       | *Generic / Various Authors*      | Improving detection rates by capturing both spatial (packet) and temporal (flow) features. | **Hybrid Framework**: Combination of **CNN** (spatial) and **LSTM/GRU** (temporal) trained on CICIDS2017/2018.                                | High computational cost makes **real-time** deployment difficult; often relies on offline analysis.                                   |
| **Ensemble Learning for Network Intrusion Detection** | *Generic / Various Authors*      | Reducing variance and improving generalizability across different attack types.            | **Ensemble Methods**: Stacking or Voting mechanisms using multiple classifiers (e.g., SVM + RF + DNN).                                        | Complex training pipelines; often tested on **imbalanced datasets** without addressing minority class failure (High False Negatives). |
| **Unsupervised Deep Learning for Zero-Day Detection** | *Generic / Various Authors*      | Detecting unknown attacks (Zero-Day) without labeled training data for them.               | **Unsupervised Learning**: Autoencoders (AE) and RBMs for anomaly detection.                                                                  | Prone to **High False Positive Rates** (classifying unusual but benign traffic as attacks).                                           |
| **Benchmark Analysis of DL Models on CICIDS2018**     | *Generic / Various Authors*      | Evaluating the effectiveness of standard DL models on modern traffic.                      | **Baseline Comparison**: Standalone CNN, RNN, and DNN architectures.                                                                          | Single models often struggle to balance **Precision vs. Recall**, detecting major attacks well but missing subtle stealthy ones.      |
| **NIDS-DL (This Project)**                            | *MusabAM*                        | **Holistic Robustness**: Accuracy, Speed, and Future-Proofing.                             | **Ensemble DL Framework**: Integrating diverse models (CNN, LSTM, Transformer) with future **Quantum** potential. Designed on **CICIDS2018**. | Currently in development; Quantum integration is in research phase due to hardware limits.                                            |

## 3. Key Comparisons and NIDS-DL Positioning

### 3.1 Dataset Relevance
Most legacy research still heavily relies on **NSL-KDD** (2009) or **UNSW-NB15** (2015). While valuable for benchmarking, these datasets do not reflect modern encrypted traffic or recent attack vectors.
*   **Contrast**: NIDS-DL primarily targets **CICIDS2018**, ensuring the system is trained on up-to-date traffic patterns, making it more relevant for real-world deployment.

### 3.2 The Ensembling Advantage
Standard projects often use single-architecture models (e.g., just a CNN). While effective for specific patterns, they lack the versatility to catch diverse attack types (e.g., DOS vs. Infiltration).
*   **Contrast**: By implementing an **Ensemble of various DL Models** (and exploring Quantum hybrids), NIDS-DL aims to achieve "Successive High Rates"—leveraging the strengths of multiple architectures to cover each other's blind spots.

### 3.3 Zero-Day and Real-Time Limits
A major gap in the literature is the trade-off between accuracy and speed. Deep, complex models (like Transformers or deep Hybrids) are accurate but slow (latency). Fast models are often less accurate.
*   **Contrast**: NIDS-DL addresses this by optimizing for **Near-Real-Time detection**, specifically targeting the "Zero-Day" challenge where speed of detection is as critical as accuracy.

## 4. Conclusion

The review of the current implementation landscape confirms that while Deep Learning is effective for NIDS, significant challenges remain in reducing false positives and achieving real-time performance on modern datasets. NIDS-DL's approach—centering on **Ensembling** and the **CICIDS2018** dataset—directly targets these industry-wide limitations, positioning it as a next-generation solution for proactive network defense.
