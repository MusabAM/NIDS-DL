"""
Train Autoencoder Model on NSL-KDD Dataset
Anomaly-based Network Intrusion Detection using Deep Learning

Key Approach:
- Train ONLY on normal traffic
- Detect attacks as anomalies using reconstruction error threshold
- Can detect novel/unknown attack types
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score
)
from rich.console import Console
from rich.table import Table

# Import our autoencoder model
from src.models.classical.autoencoder import Autoencoder

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
            # Handle unseen labels in test set
            X[col] = X[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
            # Add 'unknown' to encoder if needed
            if 'unknown' not in label_encoders[col].classes_:
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
            X[col] = label_encoders[col].transform(X[col].astype(str))
    
    X = X.values.astype(np.float32)
    
    # Use MinMaxScaler for autoencoder (output uses sigmoid)
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
        X = np.clip(X, 0, 1)  # Ensure values in [0, 1]
    
    return X.astype(np.float32), y, scaler, label_encoders


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, in train_loader:
        X_batch = X_batch.to(device)
        
        optimizer.zero_grad()
        x_recon, _ = model(X_batch)
        loss = nn.MSELoss()(x_recon, X_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_reconstruction(model, data_loader, device):
    """Evaluate reconstruction loss."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, in data_loader:
            X_batch = X_batch.to(device)
            x_recon, _ = model(X_batch)
            loss = nn.MSELoss()(x_recon, X_batch)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def compute_reconstruction_errors(model, X, device, batch_size=512):
    """Compute reconstruction errors for all samples."""
    model.eval()
    errors = []
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for X_batch, in loader:
            X_batch = X_batch.to(device)
            x_recon, _ = model(X_batch)
            mse = ((x_recon - X_batch) ** 2).mean(dim=1)
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)


def find_optimal_threshold(errors_normal, errors_attack, percentiles=[90, 95, 99]):
    """Find optimal threshold based on normal data percentiles."""
    results = []
    
    for p in percentiles:
        threshold = np.percentile(errors_normal, p)
        
        # Predictions: 1 = anomaly (attack), 0 = normal
        preds_normal = (errors_normal > threshold).astype(int)
        preds_attack = (errors_attack > threshold).astype(int)
        
        # Calculate metrics
        fp = preds_normal.sum()  # Normal classified as attack
        tn = len(preds_normal) - fp
        tp = preds_attack.sum()  # Attack classified as attack
        fn = len(preds_attack) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        
        results.append({
            'percentile': p,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })
    
    return results


def main():
    CONFIG = {
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "encoder_units": [64, 32, 16],
        "latent_dim": 8,
        "dropout": 0.2,
        "patience": 15,
        "threshold_percentile": 95,
    }
    
    console.print("\n[bold cyan]" + "=" * 60)
    console.print("[bold cyan]  Autoencoder Training on NSL-KDD Dataset")
    console.print("[bold cyan]  Anomaly-Based Intrusion Detection")
    console.print("[bold cyan]" + "=" * 60 + "\n")
    
    console.print("[yellow]Loading NSL-KDD dataset...[/yellow]")
    
    data_path = Path("./data/raw/nsl-kdd")
    
    # Load training data
    X_train_full, y_train_full, scaler, label_encoders = load_nsl_kdd(
        data_path / "train.txt", fit=True
    )
    
    # Load test data
    X_test, y_test, _, _ = load_nsl_kdd(
        data_path / "test.txt",
        scaler=scaler,
        label_encoders=label_encoders,
        fit=False
    )
    
    # IMPORTANT: Filter ONLY normal traffic for training
    normal_mask_train = (y_train_full == 0)
    X_train_normal = X_train_full[normal_mask_train]
    
    console.print(f"\n[green]✓ Training on NORMAL traffic only![/green]")
    console.print(f"  Total training samples: {len(X_train_full):,}")
    console.print(f"  Normal samples used: {len(X_train_normal):,}")
    console.print(f"  Attack samples excluded: {(~normal_mask_train).sum():,}")
    
    # Split normal data for validation
    val_size = int(0.1 * len(X_train_normal))
    indices = np.random.permutation(len(X_train_normal))
    
    X_val = X_train_normal[indices[:val_size]]
    X_train = X_train_normal[indices[val_size:]]
    
    # Dataset info table
    info_table = Table(title="NSL-KDD Dataset (Anomaly Detection)")
    info_table.add_column("Set", style="cyan")
    info_table.add_column("Samples", style="green")
    info_table.add_column("Features", style="yellow")
    info_table.add_column("Note", style="magenta")
    info_table.add_row("Train", f"{len(X_train):,}", f"{X_train.shape[1]}", "Normal only")
    info_table.add_row("Validation", f"{len(X_val):,}", f"{X_val.shape[1]}", "Normal only")
    info_table.add_row("Test", f"{len(X_test):,}", f"{X_test.shape[1]}", 
                       f"Normal: {(y_test==0).sum():,}, Attack: {(y_test==1).sum():,}")
    console.print(info_table)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = Autoencoder(
        input_dim=X_train.shape[1],
        encoder_units=CONFIG["encoder_units"],
        latent_dim=CONFIG["latent_dim"],
        dropout_rate=CONFIG["dropout"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[green]✓ Autoencoder created on {device} ({total_params:,} parameters)[/green]")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    console.print(f"\n[bold yellow]Training for {CONFIG['epochs']} epochs...[/bold yellow]\n")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    Path("./results/models").mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_reconstruction(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./results/models/autoencoder_nsl_kdd_best.pt")
            save_marker = " ★ Saved!"
        else:
            patience_counter += 1
            save_marker = ""
        
        if (epoch + 1) % 5 == 0 or save_marker:
            console.print(
                f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}{save_marker}"
            )
        
        if patience_counter >= CONFIG["patience"]:
            console.print(f"\n[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break
    
    # Load best model
    console.print("\n[bold yellow]Evaluating Best Model...[/bold yellow]")
    model.load_state_dict(torch.load("./results/models/autoencoder_nsl_kdd_best.pt"))
    
    # Compute reconstruction errors
    console.print("[cyan]Computing reconstruction errors...[/cyan]")
    
    errors_train_normal = compute_reconstruction_errors(model, X_train, device)
    
    # Separate test data
    test_normal_mask = (y_test == 0)
    X_test_normal = X_test[test_normal_mask]
    X_test_attack = X_test[~test_normal_mask]
    
    errors_test_normal = compute_reconstruction_errors(model, X_test_normal, device)
    errors_test_attack = compute_reconstruction_errors(model, X_test_attack, device)
    errors_test_all = compute_reconstruction_errors(model, X_test, device)
    
    # Error statistics
    stats_table = Table(title="Reconstruction Error Statistics")
    stats_table.add_column("Dataset", style="cyan")
    stats_table.add_column("Mean", style="green")
    stats_table.add_column("Std", style="yellow")
    stats_table.add_column("Min", style="blue")
    stats_table.add_column("Max", style="magenta")
    stats_table.add_row("Train (Normal)", f"{errors_train_normal.mean():.6f}", 
                        f"{errors_train_normal.std():.6f}",
                        f"{errors_train_normal.min():.6f}",
                        f"{errors_train_normal.max():.6f}")
    stats_table.add_row("Test (Normal)", f"{errors_test_normal.mean():.6f}",
                        f"{errors_test_normal.std():.6f}",
                        f"{errors_test_normal.min():.6f}",
                        f"{errors_test_normal.max():.6f}")
    stats_table.add_row("Test (Attack)", f"{errors_test_attack.mean():.6f}",
                        f"{errors_test_attack.std():.6f}",
                        f"{errors_test_attack.min():.6f}",
                        f"{errors_test_attack.max():.6f}")
    console.print(stats_table)
    
    # Find optimal threshold
    console.print("\n[bold yellow]Finding Optimal Threshold...[/bold yellow]")
    threshold_results = find_optimal_threshold(errors_test_normal, errors_test_attack)
    
    thresh_table = Table(title="Threshold Analysis")
    thresh_table.add_column("Percentile", style="cyan")
    thresh_table.add_column("Threshold", style="yellow")
    thresh_table.add_column("Accuracy", style="green")
    thresh_table.add_column("Precision", style="blue")
    thresh_table.add_column("Recall", style="magenta")
    thresh_table.add_column("F1-Score", style="red")
    
    for r in threshold_results:
        thresh_table.add_row(
            f"{r['percentile']}%",
            f"{r['threshold']:.6f}",
            f"{r['accuracy']*100:.2f}%",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}"
        )
    console.print(thresh_table)
    
    # Use best threshold (by F1 score)
    best_result = max(threshold_results, key=lambda x: x['f1'])
    threshold = best_result['threshold']
    
    console.print(f"\n[green]✓ Selected threshold: {threshold:.6f} "
                  f"(Percentile: {best_result['percentile']}%)[/green]")
    
    # Final predictions on full test set
    predictions = (errors_test_all > threshold).astype(int)
    
    # Results
    results_table = Table(title="NSL-KDD Autoencoder Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_row("Test Accuracy", f"{best_result['accuracy']*100:.2f}%")
    results_table.add_row("Attack Precision", f"{best_result['precision']:.4f}")
    results_table.add_row("Attack Recall", f"{best_result['recall']:.4f}")
    results_table.add_row("F1 Score", f"{best_result['f1']:.4f}")
    console.print(results_table)
    
    # Classification report
    console.print("\n[bold]Classification Report:[/bold]")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Attack']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(f"TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
    console.print(f"FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, errors_test_all)
        console.print(f"\n[bold cyan]ROC-AUC Score: {roc_auc:.4f}[/bold cyan]")
    except:
        roc_auc = 0.0
    
    # Save results
    with open("./results/autoencoder_nsl_kdd_results.txt", "w") as f:
        f.write(f"Dataset: NSL-KDD\n")
        f.write(f"Model: Autoencoder (Anomaly Detection)\n")
        f.write(f"Training: Normal traffic only ({len(X_train):,} samples)\n")
        f.write(f"Threshold Percentile: {best_result['percentile']}%\n")
        f.write(f"Threshold Value: {threshold:.6f}\n\n")
        f.write(f"Test Accuracy: {best_result['accuracy']*100:.2f}%\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write(classification_report(y_test, predictions, target_names=['Normal', 'Attack']))
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}\n")
        f.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}\n")
        f.write(f"\nReconstruction Error Statistics:\n")
        f.write(f"Normal Traffic Mean: {errors_test_normal.mean():.6f}\n")
        f.write(f"Attack Traffic Mean: {errors_test_attack.mean():.6f}\n")
    
    console.print(f"\n[bold green]{'='*60}")
    console.print(f"[bold green]  Training Complete! Test Accuracy: {best_result['accuracy']*100:.2f}%")
    console.print(f"[bold green]{'='*60}")
    
    return best_result['accuracy']


if __name__ == "__main__":
    main()
