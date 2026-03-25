import os
from pathlib import Path

import nbformat as nbf

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# 1. Title/Header
cells.append(
    nbf.v4.new_markdown_cell(
        """# Variational Quantum Classifier (VQC) using TFQ for CICIDS2018

This notebook explores the application of a Variational Quantum Classifier (VQC) to the CICIDS2018 dataset using **TensorFlow Quantum (TFQ)**.
Due to the high dimensionality of the data and the computational constraints of simulating quantum circuits on classical hardware, we will employ dimensionality reduction (PCA) and data subsampling.

**Dataset:** CICIDS2018
**Task:** Binary Classification (Benign vs. Attack)
**Frameworks:** TensorFlow Quantum (TFQ) + Cirq + TensorFlow"""
    )
)

# 2. Setup and Imports
cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Imports"))
cells.append(
    nbf.v4.new_code_cell(
        """import sys
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

# Classical ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow and TFQ tools
import tensorflow as tf
from tensorflow import keras

try:
    import cirq
    import sympy
    import tensorflow_quantum as tfq
    TFQ_AVAILABLE = True
    print("TensorFlow Quantum and Cirq imported successfully.")
except ImportError:
    TFQ_AVAILABLE = False
    print("WARNING: TensorFlow Quantum or Cirq not installed. Install with: pip install tensorflow-quantum cirq")

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Imports successful!")
"""
    )
)

# 3. Configuration
cells.append(nbf.v4.new_markdown_cell("## 2. Configuration"))
cells.append(
    nbf.v4.new_code_cell(
        """# --- Data Paths ---
DATA_DIR = project_root / 'data' / 'raw' / 'cicids2018'
RESULTS_DIR = project_root / 'results' / 'models' / 'quantum'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Quantum Parameters ---
N_QUBITS = 8             # Equal to the number of PCA components
N_LAYERS = 3             # Depth of the variational quantum circuit

# --- Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01

# --- Data Parameters ---
TOTAL_SAMPLES = 5000     # Subsample to 5k for faster execution initially

print(f"Quantum Circuit: {N_QUBITS} Qubits, {N_LAYERS} Layers")
print(f"Training: Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}")
print(f"Data: {TOTAL_SAMPLES} samples max for training+testing")
"""
    )
)

# 4. Data Loading and Preprocessing
cells.append(nbf.v4.new_markdown_cell("## 3. Data Loading and Preprocessing"))
cells.append(
    nbf.v4.new_code_cell(
        """# 3.1 Load Data
print("Loading CICIDS2018 chunks...")
csv_files = glob.glob(str(DATA_DIR / "*.csv"))

if not csv_files:
    print(f"ERROR: No CSV files found in {DATA_DIR}. Please ensure raw chunks exist.")

def load_and_sample(files, num_samples):
    \"\"\"Load chunks until we comfortably have more than `num_samples`, then sample.\"\"\"
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
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 3.2 Preprocess Data
def clean_cicids2018(df):
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    
    # Define metadata to drop
    drop_cols = [
        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 
        'Protocol', 'Timestamp'
    ]
    df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')
    
    # Handle Label
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
    print("\\nClass Distribution:\\n", df_clean['binary_label'].value_counts())
else:
    print("ERROR: binary_label not created.")
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 3.3 Splitting Data
if 'binary_label' not in df_clean.columns:
    raise ValueError("binary_label column missing from dataset.")
    
X = df_clean.drop('binary_label', axis=1).values.astype(np.float32)
y = df_clean['binary_label'].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 3.4 Scaling and PCA (Crucial for VQC)
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

# Normalize to [0, 1] range to work better with the TFQ model expectations
# Our TFQ model expects normalized features
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X_train_norm = minmax.fit_transform(X_train_pca)
X_test_norm = minmax.transform(X_test_pca)
"""
    )
)

# 5. Quantum Model Definition
cells.append(nbf.v4.new_markdown_cell("## 4. Quantum Model Definition"))
cells.append(
    nbf.v4.new_code_cell(
        """# 4.1 Import the configured TFQ Model architecture
# We utilize the predefined TFQ model classes from our quantum models source

from src.models.quantum.tfq_models import TFQClassifier, check_tfq

if TFQ_AVAILABLE:
    # Initialize TFQ Classifier
    tfq_model = TFQClassifier(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        num_classes=2,           # Binary classification
        classical_layers=[32, 16] # Postprocessing layers
    )

    # Compile the model
    # Binary classification uses binary_crossentropy under the hood
    tfq_model.compile(learning_rate=LEARNING_RATE)
    
    print(tfq_model.model.summary())
else:
    print("Cannot define TFQ model because TensorFlow Quantum is missing.")
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 4.2 Visualize Circuit
if TFQ_AVAILABLE:
    from src.models.quantum.tfq_models import visualize_circuit
    
    print("Current implementation of encoding and parameter circuits:\\n")
    visualize_circuit(tfq_model)
"""
    )
)

# 6. Training
cells.append(nbf.v4.new_markdown_cell("## 5. Training"))
cells.append(
    nbf.v4.new_code_cell(
        """# 5.1 Train the Model
if TFQ_AVAILABLE:
    print("Starting Training...")
    
    # Store training history
    history = tfq_model.fit(
        X_train=X_train_norm,
        y_train=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_norm, y_test)
    )
    
    print("Training Complete!")
"""
    )
)

# 7. Evaluation
cells.append(nbf.v4.new_markdown_cell("## 6. Evaluation"))
cells.append(
    nbf.v4.new_code_cell(
        """# 6.1 Plot Training History
if TFQ_AVAILABLE:    
    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('TFQ VQC Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('TFQ VQC Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 6.2 Classification Report
if TFQ_AVAILABLE:
    # Get predictions
    y_pred_probs = tfq_model.predict(X_test_norm)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    print("Classification Report:\\n")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 6.3 Save Model
if TFQ_AVAILABLE:
    # Save the Keras model
    model_path = RESULTS_DIR / "tfq_vqc_cicids2018.keras"
    tfq_model.model.save(model_path)
    print(f"Model saved to {model_path}")
"""
    )
)

# Add cells to notebook
nb.cells = cells

# Write notebook to file
output_path = r"c:\Users\musab\Projects\NIDS-DL\notebooks\03_quantum_dl\04_tfq_vqc_cicids2018.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook successfully created at {output_path}")
