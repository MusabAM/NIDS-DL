"""
Train Transformer Model on NSL-KDD Dataset
Network Intrusion Detection using Self-Attention

Key Features:
- Multi-head self-attention for feature relationships
- Positional encoding for feature ordering
- Global average pooling for classification
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from rich.console import Console
from rich.table import Table

# Import our transformer model
from src.models.classical.transformer import TransformerClassifier

console = Console()

# Column names for NSL-KDD
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# Binary classification mapping
ATTACK_MAPPING = {
    'normal': 0,
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
    'ipsweep': 1, 'nmap': 1, 'portsweep': 1, 'satan': 1, 'mscan': 1, 'saint': 1,
    'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1,
    'spy': 1, 'warezclient': 1, 'warezmaster': 1, 'sendmail': 1, 'named': 1,
    'snmpgetattack': 1, 'snmpguess': 1, 'xlock': 1, 'xsnoop': 1, 'worm': 1,
    'buffer_overflow': 1, 'loadmodule': 1, 'perl': 1, 'rootkit': 1,
    'httptunnel': 1, 'ps': 1, 'sqlattack': 1, 'xterm': 1,
}


def load_nsl_kdd(filepath, scaler=None, label_encoders=None, fit=True):
    """Load and preprocess NSL-KDD dataset."""
    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    
    console.print(f"[cyan]Loaded {len(df):,} samples from {filepath.name}[/cyan]")
    
    # Remove difficulty level
    df = df.drop('difficulty_level', axis=1)
    
    # Map labels to binary
    df['label'] = df['label'].map(lambda x: ATTACK_MAPPING.get(x, 1))
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label'].values
    
    # Categorical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    
    if fit:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            X[col] = X[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
            if 'unknown' not in label_encoders[col].classes_:
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
            X[col] = label_encoders[col].transform(X[col].astype(str))
    
    X = X.values.astype(np.float32)
    
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X.astype(np.float32), y, scaler, label_encoders


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return total_loss / len(data_loader), correct / total, all_preds, all_labels


def main():
    CONFIG = {
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 1e-4,
        "embed_dim": 64,
        "num_heads": 4,
        "ff_dim": 128,
        "num_blocks": 3,
        "dense_units": [64],
        "dropout": 0.3,
        "patience": 20,
    }
    
    console.print("\n[bold cyan]" + "=" * 60)
    console.print("[bold cyan]  Transformer Training on NSL-KDD Dataset")
    console.print("[bold cyan]  Self-Attention for Network Intrusion Detection")
    console.print("[bold cyan]" + "=" * 60 + "\n")
    
    console.print("[yellow]Loading NSL-KDD dataset...[/yellow]")
    
    data_path = Path("./data/raw/nsl-kdd")
    
    X_train, y_train, scaler, label_encoders = load_nsl_kdd(
        data_path / "train.txt", fit=True
    )
    
    X_test, y_test, _, _ = load_nsl_kdd(
        data_path / "test.txt",
        scaler=scaler,
        label_encoders=label_encoders,
        fit=False
    )
    
    # Validation split
    val_size = int(0.1 * len(X_train))
    indices = np.random.permutation(len(X_train))
    
    X_val = X_train[indices[:val_size]]
    y_val = y_train[indices[:val_size]]
    X_train = X_train[indices[val_size:]]
    y_train = y_train[indices[val_size:]]
    
    # Dataset info
    info_table = Table(title="NSL-KDD Dataset Information")
    info_table.add_column("Set", style="cyan")
    info_table.add_column("Samples", style="green")
    info_table.add_column("Class Distribution", style="yellow")
    info_table.add_row("Train", f"{len(X_train):,}", 
                       f"Normal: {(y_train==0).sum():,}, Attack: {(y_train==1).sum():,}")
    info_table.add_row("Validation", f"{len(X_val):,}",
                       f"Normal: {(y_val==0).sum():,}, Attack: {(y_val==1).sum():,}")
    info_table.add_row("Test", f"{len(X_test):,}",
                       f"Normal: {(y_test==0).sum():,}, Attack: {(y_test==1).sum():,}")
    console.print(info_table)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = TransformerClassifier(
        input_dim=X_train.shape[1],
        num_classes=2,
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        ff_dim=CONFIG["ff_dim"],
        num_blocks=CONFIG["num_blocks"],
        dense_units=CONFIG["dense_units"],
        dropout=CONFIG["dropout"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[green]✓ Transformer created on {device} ({total_params:,} parameters)[/green]")
    
    # Class weights
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (2 * class_counts)).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    console.print(f"\n[bold yellow]Training for {CONFIG['epochs']} epochs...[/bold yellow]\n")
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    Path("./results/models").mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        else:
            test_acc = best_test_acc
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), "./results/models/transformer_nsl_kdd_best.pt")
            save_marker = " ★ Saved!"
        else:
            patience_counter += 1
            save_marker = ""
        
        if (epoch + 1) % 5 == 0 or save_marker:
            console.print(
                f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                f"Train: {train_acc*100:.2f}% | "
                f"Val: {val_acc*100:.2f}% | "
                f"Test: {test_acc*100:.2f}%{save_marker}"
            )
        
        if patience_counter >= CONFIG["patience"]:
            console.print(f"\n[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break
    
    # Final evaluation
    console.print("\n[bold yellow]Final Evaluation on Test Set...[/bold yellow]")
    
    model.load_state_dict(torch.load("./results/models/transformer_nsl_kdd_best.pt"))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    # Results
    results_table = Table(title="NSL-KDD Transformer Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_row("Test Accuracy", f"{test_acc*100:.2f}%")
    results_table.add_row("Best Validation Accuracy", f"{best_val_acc*100:.2f}%")
    console.print(results_table)
    
    console.print("\n[bold]Classification Report:[/bold]")
    print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
    
    cm = confusion_matrix(labels, preds)
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(f"TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
    console.print(f"FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")
    
    # Save results
    with open("./results/transformer_nsl_kdd_results.txt", "w") as f:
        f.write(f"Dataset: NSL-KDD\n")
        f.write(f"Model: Transformer (Self-Attention)\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Best Validation Accuracy: {best_val_acc*100:.2f}%\n\n")
        f.write(classification_report(labels, preds, target_names=['Normal', 'Attack']))
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}\n")
        f.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}\n")
    
    console.print(f"\n[bold green]{'='*60}")
    console.print(f"[bold green]  Training Complete! Test Accuracy: {test_acc*100:.2f}%")
    console.print(f"[bold green]{'='*60}")
    
    return test_acc


if __name__ == "__main__":
    main()
