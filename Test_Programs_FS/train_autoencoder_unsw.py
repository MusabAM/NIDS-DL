"""
Semi-Supervised Autoencoder for UNSW-NB15 Dataset
Target: 79%+ accuracy using hybrid approach

Key Strategy:
- Train autoencoder primarily on normal data
- Add contrastive loss using small labeled attack samples
- This helps the model learn better separation
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
    roc_auc_score, f1_score
)
from rich.console import Console
from rich.table import Table

console = Console()

CATEGORICAL_COLS = ['proto', 'service', 'state']
DROP_COLS = ['id', 'attack_cat']
LABEL_COL = 'label'


class SemiSupervisedAutoencoder(nn.Module):
    """
    Semi-supervised autoencoder with classification head.
    Combines reconstruction + classification objectives.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], 
                 latent_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_dims = list(reversed(hidden_dims))
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout * 0.5),
            ])
            in_dim = h_dim
        
        decoder_layers.extend([
            nn.Linear(in_dim, input_dim),
            nn.Sigmoid()
        ])
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classification head from latent space
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
    
    def encode(self, x):
        x = self.input_bn(x)
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        logits = self.classifier(z)
        return x_recon, z, logits
    
    def reconstruction_error(self, x):
        with torch.no_grad():
            z = self.encode(x)
            x_recon = self.decode(z)
            return ((x - x_recon) ** 2).mean(dim=1)


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
                X[col] = X[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                X[col] = label_encoders[col].transform(X[col])
    
    X = X.fillna(0).values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
        X = np.clip(X, 0, 1)
    
    return X.astype(np.float32), y, scaler, label_encoders


def main():
    CONFIG = {
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "hidden_dims": [256, 128, 64],
        "latent_dim": 32,
        "dropout": 0.3,
        "patience": 20,
        "recon_weight": 0.3,  # Weight for reconstruction loss
        "class_weight": 0.7,  # Weight for classification loss
    }
    
    console.print("\n[bold cyan]" + "=" * 60)
    console.print("[bold cyan]  Semi-Supervised Autoencoder on UNSW-NB15")
    console.print("[bold cyan]  Target: 79%+ Accuracy")
    console.print("[bold cyan]" + "=" * 60 + "\n")
    
    data_path = Path("./data/raw/unsw-nb15")
    
    # Load ALL data including labels
    X_train_full, y_train_full, scaler, label_encoders = load_unsw_nb15(
        data_path / "training.csv", fit=True
    )
    
    X_test, y_test, _, _ = load_unsw_nb15(
        data_path / "testing.csv",
        scaler=scaler,
        label_encoders=label_encoders,
        fit=False
    )
    
    # For semi-supervised: use most normal + SOME attack samples for training
    normal_mask = (y_train_full == 0)
    attack_mask = (y_train_full == 1)
    
    X_normal = X_train_full[normal_mask]
    y_normal = y_train_full[normal_mask]
    X_attack = X_train_full[attack_mask]
    y_attack = y_train_full[attack_mask]
    
    # Use 30% of attack data for semi-supervised training
    n_attack_train = int(0.3 * len(X_attack))
    attack_indices = np.random.permutation(len(X_attack))
    X_attack_train = X_attack[attack_indices[:n_attack_train]]
    y_attack_train = y_attack[attack_indices[:n_attack_train]]
    
    # Combine for training
    X_train = np.vstack([X_normal, X_attack_train])
    y_train = np.concatenate([y_normal, y_attack_train])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    # Validation split
    val_size = int(0.1 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    # Dataset info
    info_table = Table(title="UNSW-NB15 Semi-Supervised Dataset")
    info_table.add_column("Set", style="cyan")
    info_table.add_column("Samples", style="green")
    info_table.add_column("Normal", style="blue")
    info_table.add_column("Attack", style="red")
    info_table.add_row("Train", f"{len(X_train):,}", 
                       f"{(y_train==0).sum():,}", f"{(y_train==1).sum():,}")
    info_table.add_row("Validation", f"{len(X_val):,}",
                       f"{(y_val==0).sum():,}", f"{(y_val==1).sum():,}")
    info_table.add_row("Test", f"{len(X_test):,}",
                       f"{(y_test==0).sum():,}", f"{(y_test==1).sum():,}")
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
    model = SemiSupervisedAutoencoder(
        input_dim=X_train.shape[1],
        hidden_dims=CONFIG["hidden_dims"],
        latent_dim=CONFIG["latent_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[green]✓ Model on {device} ({total_params:,} parameters)[/green]")
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (2 * class_counts)).to(device)
    
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    
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
    patience_counter = 0
    
    Path("./results/models").mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            x_recon, z, logits = model(X_batch)
            
            # Combined loss
            loss_recon = criterion_recon(x_recon, X_batch)
            loss_class = criterion_class(logits, y_batch)
            loss = CONFIG["recon_weight"] * loss_recon + CONFIG["class_weight"] * loss_class
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, _, logits = model(X_batch)
                _, predicted = logits.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "./results/models/autoencoder_unsw_nb15_semi.pt")
            save_marker = " ★ Saved!"
        else:
            patience_counter += 1
            save_marker = ""
        
        if (epoch + 1) % 5 == 0 or save_marker:
            console.print(
                f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Val Acc: {val_acc*100:.2f}%{save_marker}"
            )
        
        if patience_counter >= CONFIG["patience"]:
            console.print(f"\n[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break
    
    # Final evaluation
    console.print("\n[bold yellow]Evaluating on Test Set...[/bold yellow]")
    model.load_state_dict(torch.load("./results/models/autoencoder_unsw_nb15_semi.pt"))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            _, _, logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    test_acc = (all_preds == all_labels).mean()
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Results
    results_table = Table(title="UNSW-NB15 Semi-Supervised Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_row("Test Accuracy", f"{test_acc*100:.2f}%")
    results_table.add_row("F1 Score", f"{test_f1:.4f}")
    results_table.add_row("Best Val Accuracy", f"{best_val_acc*100:.2f}%")
    console.print(results_table)
    
    console.print("\n[bold]Classification Report:[/bold]")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Attack']))
    
    cm = confusion_matrix(all_labels, all_preds)
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(f"TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
    console.print(f"FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        console.print(f"\n[bold cyan]ROC-AUC Score: {roc_auc:.4f}[/bold cyan]")
    except:
        roc_auc = 0.0
    
    if test_acc >= 0.79:
        console.print(f"\n[bold green]🎉 TARGET ACHIEVED! Accuracy ≥ 79%[/bold green]")
    else:
        console.print(f"\n[yellow]Current: {test_acc*100:.2f}%. Target: 79%[/yellow]")
    
    # Save results
    with open("./results/autoencoder_unsw_nb15_results.txt", "w") as f:
        f.write(f"Dataset: UNSW-NB15\n")
        f.write(f"Model: Semi-Supervised Autoencoder\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=['Normal', 'Attack']))
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}\n")
        f.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}\n")
    
    console.print(f"\n[bold green]{'='*60}")
    console.print(f"[bold green]  Training Complete! Test Accuracy: {test_acc*100:.2f}%")
    console.print(f"[bold green]{'='*60}")
    
    return test_acc


if __name__ == "__main__":
    main()
