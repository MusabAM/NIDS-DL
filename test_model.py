"""
Test LSTM Model on NSL-KDD Dataset
Evaluate the trained model and generate comprehensive metrics
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.data import get_dataset, get_dataloaders
from src.models import LSTMClassifier
from src.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)

console = Console()


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    CONFIG = {
        "dataset": "nsl_kdd",
        "classification": "binary",
        "batch_size": 256,
        "model_path": "./results/models/lstm_nsl_kdd_best.pt",
    }
    
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]  LSTM Model Testing - NSL-KDD Dataset")
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
    info_table.add_row("Test Samples", f"{data['info'].test_samples:,}")
    console.print(info_table)
    
    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    _, _, test_loader = get_dataloaders(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
        batch_size=CONFIG["batch_size"],
    )
    
    # =========================================================================
    # Load Trained Model
    # =========================================================================
    console.print("\n[yellow]Loading trained model...[/yellow]")
    
    # Create model with same architecture as training
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
    
    # Load saved weights
    checkpoint = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    console.print(f"[green]✓ Model loaded from {CONFIG['model_path']}[/green]")
    console.print(f"[green]✓ Running on {device}[/green]")
    
    # =========================================================================
    # Evaluate Model
    # =========================================================================
    console.print("\n[bold yellow]Evaluating model on test set...[/bold yellow]\n")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    
    # =========================================================================
    # Compute Metrics
    # =========================================================================
    console.print("[yellow]Computing classification metrics...[/yellow]\n")
    
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=data["info"].class_names,
    )
    
    # Display results in a nice table
    results_table = Table(title="[bold]Test Results - Classification Metrics[/bold]")
    results_table.add_column("Metric", style="cyan", width=20)
    results_table.add_column("Value", style="green", width=15)
    
    results_table.add_row("Accuracy", f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    results_table.add_row("Precision", f"{metrics['precision']:.4f}")
    results_table.add_row("Recall", f"{metrics['recall']:.4f}")
    results_table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")
    
    console.print(results_table)
    
    # =========================================================================
    # Classification Report
    # =========================================================================
    console.print("\n[bold yellow]Detailed Classification Report:[/bold yellow]\n")
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=data["info"].class_names)
    console.print(report)
    
    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    console.print("\n[yellow]Generating visualizations...[/yellow]")
    
    # Create output directory for figures
    import os
    os.makedirs("./results/figures", exist_ok=True)
    
    # Confusion Matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=data["info"].class_names,
        normalize=True,
        save_path="./results/figures/confusion_matrix.png",
    )
    console.print("[green]✓ Saved confusion matrix to ./results/figures/confusion_matrix.png[/green]")
    
    # ROC Curve
    plot_roc_curve(
        y_true=y_true,
        y_proba=y_proba,
        class_names=data["info"].class_names,
        save_path="./results/figures/roc_curve.png",
    )
    console.print("[green]✓ Saved ROC curve to ./results/figures/roc_curve.png[/green]")
    
    # =========================================================================
    # Summary
    # =========================================================================
    summary_panel = Panel(
        f"""[bold green]Model Evaluation Complete![/bold green]

[cyan]Key Metrics:[/cyan]
  • Accuracy:  {metrics['accuracy']*100:.2f}%
  • Precision: {metrics['precision']*100:.2f}%
  • Recall:    {metrics['recall']*100:.2f}%
  • F1 Score:  {metrics['f1_score']*100:.2f}%

[cyan]Visualizations saved to:[/cyan]
  • ./results/figures/confusion_matrix.png
  • ./results/figures/roc_curve.png
""",
        title="[bold]Test Summary[/bold]",
        border_style="green",
    )
    console.print(summary_panel)
    
    return metrics


if __name__ == "__main__":
    main()
