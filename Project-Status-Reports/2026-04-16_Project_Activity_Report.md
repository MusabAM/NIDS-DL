# NIDS-DL Project & Activity Report
*Date: April 16, 2026*

## 1. Executive Summary
NIDS-DL is a comprehensive framework exploring both classical deep learning and emerging quantum machine learning approaches for Network Intrusion Detection Systems (NIDS). The project successfully combines traditional methods with cutting-edge hybrid models to detect network attacks while maintaining near-real-time performance.

## 2. Recent Milestones & Developments
Over our recent sessions, we have significantly extended the project's capabilities, particularly in resolving complex integration issues and pushing quantum models closer to production:

* **Batch Ensemble & Multi-Dataset Fixes:** Resolved runtime shape mismatch errors for Autoencoders and VQC models across NSL-KDD and UNSW-NB15 datasets. Successfully implemented device toggling (CPU/GPU) for inference acceleration.
* **Transformer Attack Bias Mitigation:** Eliminated attack-class bias in the Transformer model for UNSW-NB15 by correcting the Focal Loss implementation, adjusting class weights, and refactoring the data loading pipeline.
* **Quantum Model Deployment & Scaling:** Successfully deployed Hybrid Variational Quantum Classifier (VQC) models for CICIDS2018 and UNSW-NB15. Removed sampling caps, integrated PCA preprocessing transformations into the input pipeline, and added dynamic differentiation rules to support `lightning.qubit`.
* **CICIDS2017 Integration & Notebook Expansion:** Developed a dedicated Jupyter notebook for CICIDS2017 VQC to maintain consistency across the project. 
* **Frontend Enhancements:** Updated the React frontend and backend configuration to support full dataset-agnostic feature handling (CICIDS2017, NSL-KDD, etc.) for Live Prediction, Ensemble Defense, and Batch Analysis modules.

## 3. Core Project Metrics
The system has been trained and evaluated across 4 major network security datasets, achieving targets with strong performance metrics.

### Datasets Processed
- **NSL-KDD:** Legacy Benchmark (~148K samples, 41 features)
- **UNSW-NB15:** Modern Benchmark (~257K samples, 42 features)
- **CICIDS2017:** Primary Modern Dataset (2.83M samples, 77 features)
- **CICIDS2018:** Latest Modern Dataset (8.28M samples, 76 features)

### Model Variety
- **Classical DL:** CNN (1D feature extraction), LSTM (Bidirectional), Transformer (Self-Attention), Autoencoder (Unsupervised Anomaly Detection).
- **Quantum ML:** VQC (PennyLane), Hybrid Quantum-Classical architecture.

### Quantitative Results & Best Performances
| Dataset      | Best Model  | Test Acc | ROC-AUC | Precision | Recall | F1-Score |
|--------------|-------------|----------|---------|-----------|--------|----------|
| **NSL-KDD**  | Autoencoder | 85.53%   | 0.9468  | 0.86      | 0.86   | 0.87     |
| **NSL-KDD**  | VQC (Hybrid)| 77.59%   | 0.9416  | -         | -      | -        |
| **UNSW-NB15**| CNN         | 88.78%   | -       | 0.91      | 0.89   | 0.89     |
| **UNSW-NB15**| VQC (Hybrid)| 86.00%   | -       | 0.88      | 0.86   | 0.86     |
| **CICIDS2017**| CNN        | **98.00%**| -       | 0.97      | 0.97   | 0.97     |
| **CICIDS2018**| CNN        | 96.43%   | -       | 0.97      | 0.96   | 0.96     |
| **CICIDS2018**| VQC (Hybrid)| 87.00%   | -       | 0.90      | 0.87   | 0.87     |

*(Note: VQC Precision/Recall/F1-Score benchmarks reflect weighted/macro-average statistics across benign and attack traffic classes).*

> [!TIP]
> The **CNN architecture** consistently provides the best balance of speed and overall accuracy, especially on modern CICIDS datasets, easily exceeding the >95% target.
## 4. Model Training Methodologies
A consistent, robust training framework was established for both classical and quantum architectures across all datasets. Common hyperparameters include an AdamW optimizer (learning rate: 0.001), `ReduceLROnPlateau` scheduler, batch sizes of 256/512, and early stopping patience of 15-20 epochs. Inverse frequency class weighting was aggressively leveraged to handle extreme dataset imbalances (especially in CICIDS environments).

### Classical Machine Learning Models
* **1D Convolutional Neural Network (CNN):** 
  * **Design**: 1D convolutional layers with escalating filter dimensions (64, 128, 256) intended for rapid feature extraction.
  * **Training Profile**: Trained across all 4 datasets using `CrossEntropyLoss`. The CNN consistently provided the fastest inference pipeline and yielded peak accuracy scores (98.00% benchmark on CICIDS2017).
* **Bidirectional LSTM:**
  * **Design**: 3 layers featuring 256 hidden units each.
  * **Training Profile**: Targeted toward temporal pattern detection. Encountered minor overfitting issues mitigated with spatial dropout parameters (0.3 - 0.4). 
* **Transformer:**
  * **Design**: Multi-head self-attention mechanism (4 heads) generating 64-dimensional feature embeddings.
  * **Training Profile**: Leverages a corrected Focal Loss protocol interacting with specific inverted class weights. This mathematically counters inherent majoritarian attack-class biases and heavily penalizes the misclassification of minority classes (specifically targeted during the UNSW-NB15 deployment).
* **Autoencoder:**
  * **Design**: Unsupervised spatial bottleneck layer structure.
  * **Training Profile**: Trained entirely to recreate benign network flows. Anomaly detection is performed by calculating the reconstruction loss error—flagging instances breaching set thresholds as zero-day anomalies. Displayed its best independent viability on the NSL-KDD benchmark (85.53%).

### Quantum Machine Learning Models (VQC/Hybrid)
* **Variational Quantum Classifier & Hybrids:**
  * **Design**: PennyLane/PyTorch hybrid architecture weaving classic preprocessing blocks directly bounding to parametric quantum circuits.
  * **Data Pipeline (PCA Integration)**: Native network parameters (up upwards of ~77 distinct features for CICIDS) strictly breach the operational memory bounds of standard quantum operations. The VQC input pipelines inherently perform active PCA dimensionality reduction transformations—condensing network profiles strictly down to matching physical qubit sizes (e.g. reductions to 4, 8, or 16 core principal components).
  * **Training Profile**: Leverages flexible dynamic differentiation methods. Automatically calls rapid 'adjoint' differentiation protocols utilizing PennyLane's compiled `lightning.qubit` devices. Reverts mathematically to classic 'backprop' techniques purely if defaulting to legacy standard `default.qubit` state vector simulators.

## 5. Operational Readiness
* **Deployment System:** Interactive Streamlit API & frontend application for real-time predictions, batch analysis, and visual dashboarding.
* **Pipeline Infrastructure:** Full data cleaning, robust class-weighting protocols, and hardware acceleration (CUDA/PennyLane Lightning engines) are completely built and integrated.
* **Extensibility:** The backend architecture is modular, fully supporting seamless drops of newly trained checkpoint files for easy scaling.

## 6. Next Steps & Future Work
Given the current stability and functionality of the environment:
1. **Real-time Inference Expansion:** Improve the live network traffic capture streams to ensure inference latency reliably drops below 100ms.
2. **Multi-class / Threat-specific Categorization:** Extending the binary detection models to fully isolate specific cyber threats (e.g., DDoS, XSS, Port Scanning).
3. **Quantum Ensembles:** Continue optimizing PennyLane integration to allow larger multi-qubit architectures without prohibitive memory bottlenecks on local infrastructure.
