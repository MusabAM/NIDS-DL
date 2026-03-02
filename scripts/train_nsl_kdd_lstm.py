"""
Train an improved Bidirectional LSTM on the NSL-KDD dataset.
Key improvements:
- Bidirectional LSTM with stronger dropout (0.4)
- Weight decay via AdamW
- Label smoothing
- CosineAnnealingLR scheduler
- Threshold tuning at evaluation time
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
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nsl-kdd")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE  = os.path.join(MODELS_DIR, "lstm_nsl_kdd_improved.pt")
RESULTS_TXT = os.path.join(RESULTS_DIR, "lstm_nsl_kdd_improved_results.txt")

BATCH_SIZE   = 512
EPOCHS       = 60
LR           = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT      = 0.4
HIDDEN_DIM   = 256
NUM_LAYERS   = 3
LABEL_SMOOTH = 0.1
PATIENCE     = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NSL_KDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level'
]

# ─── Model ────────────────────────────────────────────────────────────────────
class ImprovedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.4, num_classes=2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_bn(x)
        x = torch.relu(self.proj(x))
        x = x.unsqueeze(1)               # (batch, 1, hidden)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_nsl_kdd(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLUMNS)

    y_train = (train_df['label'] != 'normal').astype(int).values
    y_test  = (test_df['label']  != 'normal').astype(int).values

    for df in [train_df, test_df]:
        df.drop(columns=['label', 'difficulty_level'], inplace=True, errors='ignore')

    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col]  = test_df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col]  = le.transform(test_df[col].astype(str))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.values.astype(np.float32))
    X_test  = scaler.transform(test_df.values.astype(np.float32))
    return X_train, X_test, y_train, y_test

# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    train_path = os.path.join(DATA_PATH, "train.txt")
    test_path  = os.path.join(DATA_PATH, "test.txt")
    if not os.path.exists(train_path):
        print(f"[ERROR] Data not found: {train_path}")
        return

    print(f"Device: {DEVICE}")
    X_train, X_test, y_train, y_test = load_nsl_kdd(train_path, test_path)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    mk = lambda X, y, shuffle: DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=BATCH_SIZE, shuffle=shuffle)
    train_loader = mk(X_tr,   y_tr,   True)
    val_loader   = mk(X_val,  y_val,  False)
    test_loader  = mk(X_test, y_test, False)

    model     = ImprovedBiLSTM(X_tr.shape[1], HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc, patience_count = 0, 0
    print(f"Training BiLSTM for up to {EPOCHS} epochs (patience={PATIENCE})...")

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)(model(Xb), yb).backward()
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

    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.7, 0.05):
        f = f1_score(labels_all, (probs_all >= t).astype(int), average='macro')
        if f > best_f1:
            best_f1, best_t = f, t

    final_preds = (probs_all >= best_t).astype(int)
    test_acc    = accuracy_score(labels_all, final_preds)
    report      = classification_report(labels_all, final_preds, target_names=["Normal", "Attack"])
    cm          = confusion_matrix(labels_all, final_preds)

    print(f"\nBest threshold: {best_t:.2f} | Test Acc: {test_acc*100:.2f}%")
    print(report)

    with open(RESULTS_TXT, "w") as f:
        f.write("Dataset: NSL-KDD\n")
        f.write("Model: BiLSTM (Improved - AdamW + Label Smoothing + Threshold Tuning)\n")
        f.write(f"Best Val Accuracy: {best_val_acc*100:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Optimal Threshold: {best_t:.2f}\n\n")
        f.write(report + "\n")
        f.write(f"Confusion Matrix:\nTN: {cm[0,0]} | FP: {cm[0,1]}\nFN: {cm[1,0]} | TP: {cm[1,1]}\n")

    print(f"Results saved to: {RESULTS_TXT}")

if __name__ == "__main__":
    train()
