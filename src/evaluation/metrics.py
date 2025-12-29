"""
Evaluation metrics and visualization for NIDS models.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        class_names: Names of classes (optional)
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    
    # Per-class metrics
    metrics["per_class"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    # ROC-AUC (for binary or probabilities available)
    if y_proba is not None:
        if len(np.unique(y_true)) == 2:
            # Binary classification
            if y_proba.ndim > 1:
                y_score = y_proba[:, 1]
            else:
                y_score = y_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metrics["roc_auc"] = auc(fpr, tpr)
            metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
        ax=ax,
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: Names of classes (for multi-class)
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_classes = len(np.unique(y_true))
    
    if n_classes == 2:
        # Binary classification
        if y_proba.ndim > 1:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    else:
        # Multi-class (one-vs-rest)
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            if y_proba.ndim > 1 and y_proba.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                label = class_names[i] if class_names else f'Class {i}'
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{label} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ["loss", "accuracy"],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary with metric histories
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"
        
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], label=f'Train {metric}', marker='o', markersize=3)
        
        if val_key in history:
            ax.plot(epochs, history[val_key], label=f'Val {metric}', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Training {metric.capitalize()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance: Feature importance values
        feature_names: Names of features
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importance[indices], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    output_dir: str = "./results/figures",
    experiment_name: str = "experiment",
) -> Dict[str, Any]:
    """
    Generate a complete evaluation report with metrics and figures.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        class_names: Names of classes
        output_dir: Directory for output files
        experiment_name: Name of experiment
        
    Returns:
        Dictionary with all metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    metrics = compute_classification_metrics(
        y_true, y_pred, y_proba, class_names
    )
    
    # Print summary
    print("\n" + "="*60)
    print(f"EVALUATION REPORT: {experiment_name}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("="*60 + "\n")
    
    # Generate plots
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=str(output_path / f"{experiment_name}_confusion_matrix.png")
    )
    plt.close()
    
    if y_proba is not None:
        plot_roc_curve(
            y_true, y_proba, class_names,
            save_path=str(output_path / f"{experiment_name}_roc_curve.png")
        )
        plt.close()
    
    return metrics
