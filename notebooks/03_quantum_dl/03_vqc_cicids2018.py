#!/usr/bin/env python
# coding: utf-8

# # Variational Quantum Classifier (VQC) for CICIDS2018
# 
# This notebook explores the application of a Variational Quantum Classifier (VQC) to the CICIDS2018 dataset.
# Due to the high dimensionality of the data and the computational constraints of simulating quantum circuits on classical hardware (even with GPUs), we will employ dimensionality reduction (PCA) and data subsampling.
# 
# **Dataset:** CICIDS2018
# **Task:** Binary Classification (Benign vs. Attack)
# **Frameworks:** PennyLane + PyTorch

# ## 1. Setup and Imports

# In[ ]:


import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# PennyLane for Quantum ML
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    print("WARNING: PennyLane not installed. Install with: pip install pennylane")
    PENNYLANE_AVAILABLE = False

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Imports successful!")


# In[ ]:


# Set up device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if PENNYLANE_AVAILABLE:
    print("\nPennyLane is ready for quantum device simulation.")


# ## 2. Configuration

# In[ ]:


# --- Data Paths ---
DATA_DIR = project_root / 'data' / 'raw' / 'cicids2018'
RESULTS_DIR = project_root / 'results' / 'models' / 'quantum'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Quantum Parameters ---
N_QUBITS = 8             # Equal to the number of PCA components to simplify angle embedding
N_LAYERS = 4             # Depth of the StronglyEntanglingLayers

# --- Training Parameters ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01     # Often needs to be slightly higher for VQCs

# --- Data Parameters ---
TOTAL_SAMPLES = 5000     # Subsample to 5k for faster execution initially

print(f"Quantum Circuit: {N_QUBITS} Qubits, {N_LAYERS} Layers")
print(f"Training: Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}")
print(f"Data: {TOTAL_SAMPLES} samples max for training+testing")


# ## 3. Data Loading and Preprocessing

# In[ ]:


# 3.1 Load Data
print("Loading CICIDS2018 chunks...")
csv_files = glob.glob(str(DATA_DIR / "*.csv"))

if not csv_files:
    print(f"ERROR: No CSV files found in {DATA_DIR}. Please ensure raw chunks exist.")

def load_and_sample(files, num_samples):
    """Load chunks until we comfortably have more than `num_samples`, then sample."""
    dfs = []
    total_loaded = 0
    for file in files:
        try:
            df_chunk = pd.read_csv(file, nrows=num_samples) 
            dfs.append(df_chunk)
            total_loaded += len(df_chunk)
            if total_loaded >= num_samples * 2:
                break
        except Exception as e:
            print(f"Skipping {os.path.basename(file)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined.sample(n=min(num_samples, len(combined)), random_state=SEED)

df_raw = load_and_sample(csv_files, TOTAL_SAMPLES)
print(f"Loaded {len(df_raw)} raw samples.")


# In[ ]:


# 3.2 Preprocess Data
def clean_cicids2018(df):
    df_clean = df.copy()

    # Standardize column names to prevent matching issues
    df_clean.columns = [str(c).strip() for c in df_clean.columns]

    # Define metadata to drop
    drop_cols = [
        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 
        'Protocol', 'Timestamp'
    ]
    df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')

    # Handle Label (case-insensitive search just in case)
    label_col = None
    for col in df_clean.columns:
        if col.lower() == 'label':
            label_col = col
            break

    if label_col is not None:
        df_clean['binary_label'] = df_clean[label_col].apply(lambda x: 0 if x == 'Benign' else 1)
        df_clean = df_clean.drop(label_col, axis=1)
    else:
        print("WARNING: Label column not found.")

    # Handle Inf/NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()

    return df_clean

df_clean = clean_cicids2018(df_raw)
print(f"Cleaned samples (remaining): {len(df_clean)}")
if 'binary_label' in df_clean.columns:
    print("\nClass Distribution:\n", df_clean['binary_label'].value_counts())
else:
    print("ERROR: binary_label not created.")


# In[ ]:


# 3.3 Splitting Data
if 'binary_label' not in df_clean.columns:
    raise ValueError("binary_label column missing from dataset.")

X = df_clean.drop('binary_label', axis=1).values.astype(np.float32)
y = df_clean['binary_label'].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")


# In[ ]:


# 3.4 Scaling and PCA (Crucial for VQC)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Original feature space size: {X_train_scaled.shape[1]}")

# Apply PCA to reduce features to N_QUBITS
pca = PCA(n_components=N_QUBITS)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Reduced feature space size (via PCA): {X_train_pca.shape[1]}")
print(f"Explained Variance Ratio (total): {sum(pca.explained_variance_ratio_):.4f}")


# In[ ]:


# 3.5 Create PyTorch Datasets/Loaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")


# ## 4. Quantum Model Definition

# In[ ]:


# 4.1 Define the Quantum Circuit using PennyLane
if PENNYLANE_AVAILABLE:
    dev = qml.device('lightning.qubit', wires=N_QUBITS)
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

    @qml.qnode(dev, interface='torch')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))
else:
    print("PennyLane missing. Model building will fail below unless manually patched.")
    weight_shapes = None
    quantum_circuit = None


# In[ ]:


# 4.2 Define Hybrid PyTorch Module
class HybridQuantumVQC(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super().__init__()
        if PENNYLANE_AVAILABLE:
            self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        else:
            self.qlayer = nn.Linear(N_QUBITS, 1) # simple fallback

    def forward(self, x):
        if PENNYLANE_AVAILABLE:
            q_out = self.qlayer(x)
        else:
            q_out = torch.tanh(self.qlayer(x)).squeeze() # Fallback mapping to [-1, 1]

        # Transform scalar q_out to binary logits: 
        output_logits = torch.cat([
            (1 - q_out).unsqueeze(1) / 2.0, 
            (1 + q_out).unsqueeze(1) / 2.0  
        ], dim=1)
        return output_logits

# Initialize hybrid model
model = HybridQuantumVQC(quantum_circuit, weight_shapes).to(device)
print(model)


# ## 5. Training

# In[ ]:


def custom_loss(outputs, labels):
    outputs = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)
    return nn.NLLLoss()(torch.log(outputs), labels)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[ ]:


print("Starting Training...")
train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for i, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = custom_loss(outputs, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_train_loss:.4f} - Val Accuracy: {val_acc*100:.2f}%")

print("Training Complete!")


# ## 6. Evaluation

# In[ ]:


# Plot the training loss and validation accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('VQC Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.title('VQC Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
# plt.show()


# In[ ]:


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=['Benign', 'Attack']))


# In[ ]:


# Save the trained weights
model_path = RESULTS_DIR / "vqc_cicids2018.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

