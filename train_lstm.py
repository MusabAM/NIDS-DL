"""
LSTM Model Training on NSL-KDD Dataset
Network Intrusion Detection using Deep Learning
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data import get_dataset, get_dataloaders
from src.models import LSTMClassifier
from src.training import TorchTrainer

console = Console()

def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    CONFIG = {
        "dataset": "nsl_kdd",
        "classification": "binary",  # binary or multiclass
        "batch_size": 256,
        "epochs": 30,
        "learning_rate": 0.001,
        "early_stopping_patience": 10,
    }
    
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]  LSTM Model Training - NSL-KDD Dataset")
    console.print("[bold cyan]=" * 60 + "\n")
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    console.print("[yellow]Loading NSL-KDD dataset...[/yellow]")
    
    data = get_dataset(
        name=CONFIG["dataset"],
        classification=CONFIG["classification"],
        normalize="standard",
        handle_imbalance="none",
    )
    
    # Print dataset info
    info_table = Table(title="Dataset Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Dataset", data["info"].name.upper())
    info_table.add_row("Features", str(data["info"].num_features))
    info_table.add_row("Classes", str(data["info"].num_classes))
    info_table.add_row("Class Names", str(data["info"].class_names))
    info_table.add_row("Train Samples", f"{data['info'].train_samples:,}")
    info_table.add_row("Validation Samples", f"{data['info'].val_samples:,}")
    info_table.add_row("Test Samples", f"{data['info'].test_samples:,}")
    console.print(info_table)
    
    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    train_loader, val_loader, test_loader = get_dataloaders(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
        batch_size=CONFIG["batch_size"],
    )
    
    # =========================================================================
    # Create LSTM Model
    # =========================================================================
    console.print("\n[yellow]Creating LSTM model...[/yellow]")
    
    model = LSTMClassifier(
        input_dim=data["info"].num_features,
        num_classes=data["info"].num_classes,
        lstm_units=[128, 64],
        dense_units=[128, 64],
        bidirectional=True,
        dropout_rate=0.3,
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    console.print(f"[green]✓ Model created on {device}[/green]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]  Total parameters: {total_params:,}[/dim]")
    console.print(f"[dim]  Trainable parameters: {trainable_params:,}[/dim]")
    
    # =========================================================================
    # Training Setup
    # =========================================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    trainer = TorchTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_name="lstm_nsl_kdd",
    )
    
    # =========================================================================
    # Train the Model
    # =========================================================================
    console.print(f"\n[bold yellow]Starting training for {CONFIG['epochs']} epochs...[/bold yellow]\n")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG["epochs"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        save_best=True,
        model_path="./results/models/lstm_nsl_kdd_best.pt",
    )
    
    # =========================================================================
    # Evaluate on Test Set
    # =========================================================================
    console.print("\n[bold yellow]Evaluating on test set...[/bold yellow]")
    
    results = trainer.evaluate(test_loader)
    
    # Print results
    results_table = Table(title="Test Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_row("Accuracy", f"{results['test_accuracy']:.4f}")
    results_table.add_row("Loss", f"{results['test_loss']:.4f}")
    console.print(results_table)
    
    # =========================================================================
    # Summary
    # =========================================================================
    console.print("\n[bold green]=" * 60)
    console.print("[bold green]  Training Complete!")
    console.print("[bold green]=" * 60)
    console.print(f"\n[dim]Model saved to: ./results/models/lstm_nsl_kdd_best.pt[/dim]")
    
    return results

if __name__ == "__main__":
    main()
