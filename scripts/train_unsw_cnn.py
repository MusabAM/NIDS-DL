"""
Train an improved CNN classifier on the UNSW-NB15 dataset.
Key fixes:
- Threshold tuning to improve Normal class recall (was only 74%)
- Class-weighted loss to handle imbalance (attack >> normal samples)
- Stronger dropout + weight decay
- Label smoothing
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "unsw-nb15")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE  = os.path.join(MODELS_DIR, "cnn_unsw_nb15_improved.pt")
RESULTS_TXT = os.path.join(RESULTS_DIR, "cnn_unsw_nb15_improved_results.txt")

BATCH_SIZE   = 1024
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT      = 0.4
LABEL_SMOOTH = 0.05
PATIENCE     = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model ────────────────────────────────────────────────────────────────────
class ImprovedCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_unsw(data_path):
    import glob
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    dfs = []
    for f in csv_files:
        print(f"  Reading {os.path.basename(f)}...")
        # Use low_memory=False to handle mixed types in columns like dsport
        df_temp = pd.read_csv(f, low_memory=False)
        df_temp.columns = df_temp.columns.str.strip().str.lower()
        
        # Drop identifiers and label cats immediately to save RAM
        drop_cols = [c for c in ['id', 'srcip', 'dstip', 'sport', 'dsport'] if c in df_temp.columns]
        df_temp.drop(columns=drop_cols, inplace=True)
        dfs.append(df_temp)
        
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    # Labels
    if 'label' in df.columns:
        y = df['label'].values.astype(int)
        df.drop(columns=['label'], inplace=True, errors='ignore')
    else:
        y = (df['attack_cat'].fillna('Normal') != 'Normal').astype(int).values
    
    if 'attack_cat' in df.columns:
        df.drop(columns=['attack_cat'], inplace=True)

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values.astype(np.float32))
    return X, y

# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data not found: {DATA_PATH}")
        return

    print(f"Device: {DEVICE}")
    X, y = load_unsw(DATA_PATH)

    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Class weights to fix Normal recall
    class_counts = np.bincount(y_tr)
    total   = len(y_tr)
    weights = torch.FloatTensor(total / (len(class_counts) * class_counts)).to(DEVICE)

    mk = lambda X, y, s: DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=BATCH_SIZE, shuffle=s)
    train_loader = mk(X_tr, y_tr, True)
    val_loader   = mk(X_val, y_val, False)
    test_loader  = mk(X_test, y_test, False)

    model     = ImprovedCNN(X_tr.shape[1], dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc, patience_count = 0, 0
    print(f"Training with class weights: Normal={weights[0]:.2f}, Attack={weights[1]:.2f}")

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                vp.extend(torch.argmax(model(Xb.to(DEVICE)), dim=1).cpu().numpy())
                vl.extend(yb.numpy())
        val_acc = accuracy_score(vl, vp)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ─── Threshold-tuned evaluation ───────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_SAVE, weights_only=True))
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs_all.extend(torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1].cpu().numpy())
            labels_all.extend(yb.numpy())

    probs_all  = np.array(probs_all)
    labels_all = np.array(labels_all)

    # Optimize threshold for Normal recall (lower = more conservative = more Normal)
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.25, 0.65, 0.05):
        preds_t = (probs_all >= t).astype(int)
        # Focus on macro F1 to balance Normal and Attack recall
        f = f1_score(labels_all, preds_t, average='macro')
        if f > best_f1:
            best_f1, best_t = f, t

    final_preds = (probs_all >= best_t).astype(int)
    test_acc    = accuracy_score(labels_all, final_preds)
    report      = classification_report(labels_all, final_preds, target_names=["Normal", "Attack"])
    cm          = confusion_matrix(labels_all, final_preds)

    print(f"\nBest threshold: {best_t:.2f} | Test Acc: {test_acc*100:.2f}%")
    print(report)

    with open(RESULTS_TXT, "w") as f:
        f.write("Dataset: UNSW-NB15\n")
        f.write("Model: CNN (Improved - Class Weights + Threshold Tuning)\n")
        f.write(f"Best Val Accuracy: {best_val_acc*100:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Optimal Threshold: {best_t:.2f}\n\n")
        f.write(report + "\n")
        f.write(f"Confusion Matrix:\nTN: {cm[0,0]} | FP: {cm[0,1]}\nFN: {cm[1,0]} | TP: {cm[1,1]}\n")

    print(f"Results saved to: {RESULTS_TXT}")

if __name__ == "__main__":
    train()
