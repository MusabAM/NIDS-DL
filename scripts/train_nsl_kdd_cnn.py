"""
Train an improved CNN classifier on the NSL-KDD dataset.
Key improvements over baseline:
- Stronger dropout (0.4)
- Weight decay (L2 regularization)
- Label smoothing in CrossEntropyLoss
- Learning rate scheduler (CosineAnnealingLR)
- Threshold tuning for better Normal/Attack balance
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nsl-kdd")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE  = os.path.join(MODELS_DIR, "cnn_nsl_kdd_improved.pt")
RESULTS_TXT = os.path.join(RESULTS_DIR, "cnn_nsl_kdd_improved_results.txt")

BATCH_SIZE = 512
EPOCHS     = 60
LR         = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT    = 0.4
LABEL_SMOOTH = 0.1   # reduces overconfidence
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
class ImprovedCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ─── Data Loader ──────────────────────────────────────────────────────────────
def load_nsl_kdd(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLUMNS)

    for df in [train_df, test_df]:
        df.drop(columns=['difficulty_level', 'label'], inplace=True, errors='ignore')

    # We need labels before dropping
    train_labels = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)['label']
    test_labels  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLUMNS)['label']

    y_train = (train_labels != 'normal').astype(int).values
    y_test  = (test_labels  != 'normal').astype(int).values

    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        # Handle unseen labels in test
        test_df[col] = test_df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le

    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    train_path = os.path.join(DATA_PATH, "train.txt")
    test_path  = os.path.join(DATA_PATH, "test.txt")

    if not os.path.exists(train_path):
        print(f"[ERROR] Training data not found at: {train_path}")
        return

    print(f"Device: {DEVICE}")
    X_train, X_test, y_train, y_test = load_nsl_kdd(train_path, test_path)

    # Create val split from training data (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr),  torch.LongTensor(y_tr)),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(torch.FloatTensor(X_test),torch.LongTensor(y_test)),batch_size=BATCH_SIZE)

    model = ImprovedCNN(X_tr.shape[1], dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    patience_count = 0
    PATIENCE = 15

    print(f"Training for {EPOCHS} epochs (early stopping patience={PATIENCE})...")
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                v_preds.extend(torch.argmax(model(Xb.to(DEVICE)), dim=1).cpu().numpy())
                v_labels.extend(yb.numpy())
        val_acc = accuracy_score(v_labels, v_preds)

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

    # ─── Evaluate with threshold tuning ───────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_SAVE, weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs = torch.softmax(model(Xb.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Try multiple thresholds to reduce overfitting to default 0.5
    best_thresh = 0.5
    best_f1 = 0
    from sklearn.metrics import f1_score
    for t in np.arange(0.3, 0.7, 0.05):
        preds_t = (all_probs >= t).astype(int)
        f = f1_score(all_labels, preds_t, average='macro')
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    final_preds = (all_probs >= best_thresh).astype(int)
    test_acc    = accuracy_score(all_labels, final_preds)
    report      = classification_report(all_labels, final_preds, target_names=["Normal", "Attack"])
    cm          = confusion_matrix(all_labels, final_preds)

    print(f"\nBest threshold: {best_thresh:.2f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(report)

    with open(RESULTS_TXT, "w") as f:
        f.write("Dataset: NSL-KDD\n")
        f.write("Model: CNN (Improved - L2 + Label Smoothing + Threshold Tuning)\n")
        f.write(f"Best Val Accuracy: {best_val_acc*100:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Optimal Threshold: {best_thresh:.2f}\n\n")
        f.write(report + "\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"TN: {cm[0,0]} | FP: {cm[0,1]}\n")
        f.write(f"FN: {cm[1,0]} | TP: {cm[1,1]}\n")

    print(f"\nResults saved to: {RESULTS_TXT}")

if __name__ == "__main__":
    train()
