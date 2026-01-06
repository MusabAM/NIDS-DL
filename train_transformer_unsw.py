"""
Train Transformer Model on UNSW-NB15 Dataset
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

# UNSW-NB15 column info
CATEGORICAL_COLS = ['proto', 'service', 'state']
DROP_COLS = ['id', 'attack_cat']
LABEL_COL = 'label'


def load_unsw_nb15(filepath, scaler=None, label_encoders=None, fit=True):
    """Load and preprocess UNSW-NB15 dataset."""
    df = pd.read_csv(filepath)
    
    console.print(f"[cyan]Loaded {len(df):,} samples from {filepath.name}[/cyan]")
    
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    
    y = df[LABEL_COL].values
    X = df.drop(columns=[LABEL_COL])
    
    if fit:
        label_encoders = {}
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = X[col].fillna('unknown').astype(str)
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].fillna('unknown').astype(str)
                X[col] = X[col].apply(
                    lambda x: x if x in label_encoders[col].classes_ else 'unknown'
                )
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(
                        label_encoders[col].classes_, 'unknown'
                    )
                X[col] = label_encoders[col].transform(X[col])
    
    X = X.fillna(0)
    X = X.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
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
        "patience": 15,
    }
    
    console.print("\n[bold cyan]" + "=" * 60)
    console.print("[bold cyan]  Transformer Training on UNSW-NB15 Dataset")
    console.print("[bold cyan]  Self-Attention for Network Intrusion Detection")
    console.print("[bold cyan]" + "=" * 60 + "\n")
    
    console.print("[yellow]Loading UNSW-NB15 dataset...[/yellow]")
    
    data_path = Path("./data/raw/unsw-nb15")
    
    X_train, y_train, scaler, label_encoders = load_unsw_nb15(
        data_path / "training.csv", fit=True
    )
    
    X_test, y_test, _, _ = load_unsw_nb15(
        data_path / "testing.csv",
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
    info_table = Table(title="UNSW-NB15 Dataset Information")
    info_table.add_column("Set", style="cyan")
    info_table.add_column("Samples", style="green")
    info_table.add_column("Features", style="yellow")
    info_table.add_row("Train", f"{len(X_train):,}", f"{X_train.shape[1]}")
    info_table.add_row("Validation", f"{len(X_val):,}", f"{X_val.shape[1]}")
    info_table.add_row("Test", f"{len(X_test):,}", f"{X_test.shape[1]}")
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
    
    # Class weights for imbalanced data
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
            torch.save(model.state_dict(), "./results/models/transformer_unsw_nb15_best.pt")
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
    
    model.load_state_dict(torch.load("./results/models/transformer_unsw_nb15_best.pt"))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    # Results
    results_table = Table(title="UNSW-NB15 Transformer Results")
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
    with open("./results/transformer_unsw_nb15_results.txt", "w") as f:
        f.write(f"Dataset: UNSW-NB15\n")
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
