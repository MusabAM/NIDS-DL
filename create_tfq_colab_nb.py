import os
from pathlib import Path

import nbformat as nbf

# Create a new notebook object
nb = nbf.v4.new_notebook()
cells = []

# 1. Title/Header
cells.append(
    nbf.v4.new_markdown_cell(
        """# TFQ Variational Quantum Classifier (Colab)

This notebook implements a Variational Quantum Classifier (VQC) using **TensorFlow Quantum (TFQ)** on the CICIDS2018 dataset.
It is optimized to run on **Google Colab** and fetches data directly from Google Drive.

**Dataset:** CICIDS2018
**Task:** Binary Classification (Benign vs. Attack)
**Frameworks:** TensorFlow Quantum (TFQ) + Cirq + TensorFlow

## Setup Instructions
1. **Enable GPU**: `Runtime → Change runtime type → T4 GPU`
2. **Upload data to Google Drive** at `MyDrive/NIDS-DL/data/raw/cicids2018/` (CSV files)
3. **Run cells top-to-bottom**
"""
    )
)

# 2. Installs
cells.append(nbf.v4.new_markdown_cell("## 0. Install Dependencies & Mount Drive"))
cells.append(
    nbf.v4.new_code_cell(
        """# Install TFQ and Cirq
!pip install tensorflow-quantum cirq --quiet
print("Dependencies installed.")
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Check GPU availability
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ GPU is available!")
    print(result.stdout[:500])
else:
    print("❌ No GPU detected. Go to Runtime → Change runtime type → select 'T4 GPU' or suitable option.")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("Google Drive mounted at /content/drive")
"""
    )
)

# 3. Setup and Imports
cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Imports"))
cells.append(
    nbf.v4.new_code_cell(
        """import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# Quantum tools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cirq
import sympy
import tensorflow_quantum as tfq

# Set style & seeds
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Imports successful!")
"""
    )
)

# 4. Configuration
cells.append(nbf.v4.new_markdown_cell("## 2. Configuration"))
cells.append(
    nbf.v4.new_code_cell(
        """# --- Data Paths (Google Drive) ---
DATA_DIR = '/content/drive/MyDrive/NIDS-DL/data/raw/cicids2018/'
RESULTS_DIR = '/content/drive/MyDrive/NIDS-DL/results/models/quantum/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Quantum Parameters ---
N_QUBITS = 8             # Equal to the number of PCA components
N_LAYERS = 3             # Depth of the variational quantum circuit

# --- Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01

# --- Data Parameters ---
TOTAL_SAMPLES = 5000     # Adjust based on Colab limits and time available
MAX_FILE_SIZE_MB = 1000

print(f"Data Dir: {DATA_DIR}")
print(f"Results Dir: {RESULTS_DIR}")
print(f"Quantum Circuit: {N_QUBITS} Qubits, {N_LAYERS} Layers")
"""
    )
)

# 5. Data Loading
cells.append(nbf.v4.new_markdown_cell("## 3. Data Loading and Preprocessing"))
cells.append(
    nbf.v4.new_code_cell(
        """# 3.1 Load Data
print("Loading CICIDS2018 chunks...")
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

if not all_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Please upload CICIDS2018 CSV files to Google Drive.")

def load_and_sample(files, num_samples):
    dfs = []
    total_loaded = 0
    for file in files:
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            continue
        try:
            df_chunk = pd.read_csv(file, nrows=num_samples, low_memory=False) 
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

df_raw = load_and_sample(all_files, TOTAL_SAMPLES)
print(f"Loaded {len(df_raw)} raw samples.")
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 3.2 Preprocess Data
def clean_cicids2018(df):
    df_clean = df.copy()
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    
    drop_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp']
    df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], inplace=True, errors='ignore')
    
    label_col = None
    for col in df_clean.columns:
        if col.lower() == 'label':
            label_col = col
            break
            
    if label_col is not None:
        df_clean['binary_label'] = df_clean[label_col].apply(lambda x: 0 if isinstance(x, str) and 'BENIGN' in x.upper() else 1)
        df_clean.drop(label_col, axis=1, inplace=True)
        
    feature_cols = [c for c in df_clean.columns if c != 'binary_label']
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    
    return df_clean

df_clean = clean_cicids2018(df_raw)
print(f"Samples after cleaning: {len(df_clean)}")
print("\\nLabel distribution:")
print(df_clean['binary_label'].value_counts())
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# 3.3 Splitting and Dimensionality Reduction (PCA)
X = df_clean.drop('binary_label', axis=1).values.astype(np.float32)
y = df_clean['binary_label'].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=N_QUBITS)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Reduced feature space size (via PCA): {X_train_pca.shape[1]}")

# Normalize to [0, 1] for angle encoding in the quantum circuit
minmax = MinMaxScaler()
X_train_norm = minmax.fit_transform(X_train_pca)
X_test_norm = minmax.transform(X_test_pca)
"""
    )
)

# 6. TFQ Model Definition (Inlined)
cells.append(
    nbf.v4.new_markdown_cell(
        "## 4. Quantum Model Definition (Standalone TFQ Classifier)"
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """# Since we are running on Colab without fetching the whole source codebase, 
# we inline the necessary Cirq and TFQ Model building functions here.

def create_encoding_circuit(qubits, input_symbols):
    circuit = cirq.Circuit()
    for qubit, symbol in zip(qubits, input_symbols):
        circuit.append(cirq.ry(symbol * np.pi).on(qubit))
    return circuit

def create_variational_circuit(qubits, n_layers, parameter_prefix="θ"):
    n_qubits = len(qubits)
    circuit = cirq.Circuit()
    symbols = []
    symbol_idx = 0
    for layer in range(n_layers):
        for i, qubit in enumerate(qubits):
            for gate_name, gate in [("rx", cirq.rx), ("ry", cirq.ry), ("rz", cirq.rz)]:
                symbol = sympy.Symbol(f"{parameter_prefix}_{layer}_{i}_{gate_name}")
                symbols.append(symbol)
                circuit.append(gate(symbol).on(qubit))
                symbol_idx += 1
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        if n_qubits > 1:
            circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    return circuit, symbols

def create_readout_operators(qubits, n_outputs):
    operators = []
    for i in range(min(n_outputs, len(qubits))):
        operators.append(cirq.Z(qubits[i]))
    return operators

class TFQClassifier:
    def __init__(self, n_qubits=8, n_layers=3, num_classes=2, classical_layers=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.classical_layers = classical_layers or [32, 16]
        
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.input_symbols = [sympy.Symbol(f"x_{i}") for i in range(n_qubits)]
        
        self.encoding_circuit = create_encoding_circuit(self.qubits, self.input_symbols)
        self.var_circuit, self.var_symbols = create_variational_circuit(self.qubits, n_layers)
        
        self.readout_ops = create_readout_operators(self.qubits, num_classes)
        self.model = self._build_model()

    def _build_model(self):
        circuit_input = keras.Input(shape=(), dtype=tf.string, name="circuits")
        pqc = tfq.layers.PQC(
            self.var_circuit,
            self.readout_ops,
            repetitions=1000,
            differentiator=tfq.differentiators.ParameterShift(),
        )
        x = pqc(circuit_input)
        for units in self.classical_layers:
            x = layers.Dense(units, activation="relu")(x)
            
        if self.num_classes == 2:
            output = layers.Dense(1, activation="sigmoid")(x)
        else:
            output = layers.Dense(self.num_classes, activation="softmax")(x)
            
        return keras.Model(inputs=circuit_input, outputs=output)

    def preprocess_data(self, X):
        circuits = []
        for sample in X:
            resolver = cirq.ParamResolver({str(sym): val for sym, val in zip(self.input_symbols, sample[: self.n_qubits])})
            resolved_circuit = cirq.resolve_parameters(self.encoding_circuit, resolver)
            circuits.append(resolved_circuit)
        return tfq.convert_to_tensor(circuits)

    def compile(self, learning_rate=0.01, loss=None):
        if loss is None:
            loss = "binary_crossentropy" if self.num_classes == 2 else "sparse_categorical_crossentropy"
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"])

    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None, **kwargs):
        X_train_q = self.preprocess_data(X_train)
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_q = self.preprocess_data(X_val)
            val_data = (X_val_q, y_val)
        return self.model.fit(X_train_q, y_train, epochs=epochs, batch_size=batch_size, validation_data=val_data, **kwargs)

    def predict(self, X):
        X_q = self.preprocess_data(X)
        return self.model.predict(X_q)
        
# Instantiate the model
tfq_model = TFQClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS, num_classes=2, classical_layers=[32, 16])
tfq_model.compile(learning_rate=LEARNING_RATE)
print(tfq_model.model.summary())
"""
    )
)

# 7. Training
cells.append(nbf.v4.new_markdown_cell("## 5. Training"))
cells.append(
    nbf.v4.new_code_cell(
        """print("Starting Training...")
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

# 8. Evaluation
cells.append(nbf.v4.new_markdown_cell("## 6. Evaluation"))
cells.append(
    nbf.v4.new_code_cell(
        """# Plot Training History
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('TFQ VQC Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

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
        """# Classification Report
y_pred_probs = tfq_model.predict(X_test_norm)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("Classification Report:\\n")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))

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
        """# Save Model
model_path = os.path.join(RESULTS_DIR, "tfq_vqc_cicids2018.keras")
tfq_model.model.save(model_path)
print(f"Model saved to {model_path}")
"""
    )
)

nb.cells = cells
output_path = r"c:\Users\musab\Projects\NIDS-DL\notebooks\03_quantum_dl\05_tfq_vqc_cicids2018_colab.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Colab Notebook successfully created at {output_path}")
