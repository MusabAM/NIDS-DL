"""
Dataset classes and data loaders for PyTorch and TensorFlow.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, Literal
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

try:
    import tensorflow as tf
except ImportError:
    tf = None

from rich.console import Console

console = Console()


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    num_features: int
    num_classes: int
    class_names: list
    train_samples: int
    val_samples: int
    test_samples: int


class NIDSDataset(Dataset):
    """
    PyTorch Dataset for NIDS data.
    
    Attributes:
        X: Feature tensor
        y: Label tensor
        transform: Optional transform function
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None,
    ):
        """
        Initialize NIDS Dataset.
        
        Args:
            X: Features as numpy array
            y: Labels as numpy array
            transform: Optional transform function
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (for GPU)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = NIDSDataset(X_train, y_train)
    val_dataset = NIDSDataset(X_val, y_val)
    test_dataset = NIDSDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    console.print(f"[green]✓[/green] Created DataLoaders (batch_size={batch_size})")
    
    return train_loader, val_loader, test_loader


def get_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    prefetch: bool = True,
) -> Tuple["tf.data.Dataset", "tf.data.Dataset", "tf.data.Dataset"]:
    """
    Create TensorFlow Datasets for train, validation, and test sets.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size for training
        prefetch: Whether to prefetch data
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    if tf is None:
        raise ImportError("TensorFlow is required for get_tf_datasets")
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Shuffle and batch training data
    train_ds = train_ds.shuffle(buffer_size=10000).batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    if prefetch:
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    console.print(f"[green]✓[/green] Created TensorFlow Datasets (batch_size={batch_size})")
    
    return train_ds, val_ds, test_ds


def get_dataset(
    name: Literal["nsl_kdd", "cicids2017", "cicids2018", "unsw_nb15"],
    data_dir: str = "./data",
    classification: str = "binary",
    normalize: str = "standard",
    handle_imbalance: str = "none",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    feature_engineering: bool = False,
    feature_selection: bool = False,
    k_features: Union[int, float] = 0.8,
) -> Dict[str, Any]:
    """
    Load and preprocess a complete dataset.
    
    Args:
        name: Dataset name
        data_dir: Data directory
        classification: 'binary' or 'multiclass'
        normalize: Normalization method
        handle_imbalance: Resampling method
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        feature_engineering: Whether to apply feature engineering (log transform)
        feature_selection: Whether to apply feature selection
        k_features: Number/proportion of features to keep
        
    Returns:
        Dictionary with processed data and metadata
    """
    from .preprocessing import (
        preprocess_nsl_kdd,
        preprocess_cicids,
        preprocess_unsw_nb15,
        encode_categorical,
        normalize_features,
        encode_labels,
        handle_class_imbalance,
        split_data,
        feature_engineering as apply_eng,
        select_features as apply_sel,
    )
    
    data_path = Path(data_dir) / "raw"
    
    # Load dataset
    if name == "nsl_kdd":
        train_path = data_path / "nsl-kdd" / "train.txt"
        X, y = preprocess_nsl_kdd(train_path, classification)
    elif name in ["cicids2017", "cicids2018"]:
        dataset_path = data_path / name
        X, y = preprocess_cicids(dataset_path, classification)
    elif name == "unsw_nb15":
        dataset_path = data_path / "unsw-nb15"
        X, y = preprocess_unsw_nb15(dataset_path, classification)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Feature Engineering (Log Transformation)
    if feature_engineering:
        X = apply_eng(X)
    
    # Encode labels
    y_encoded, label_encoder = encode_labels(y)
    
    # Encode categorical features (Requirement for Mutual Information)
    X, cat_encoders = encode_categorical(X, method="label")
    
    # Feature Selection
    selected_features = X.columns.tolist()
    if feature_selection:
        X, selected_features = apply_sel(X, y_encoded, k=k_features)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X.values, y_encoded,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    
    # Handle class imbalance (only on training data)
    if handle_imbalance != "none":
        X_train, y_train = handle_class_imbalance(
            X_train, y_train,
            method=handle_imbalance,
            random_state=random_state,
        )
    
    # Normalize features
    X_train, X_val, scaler = normalize_features(X_train, X_val, method=normalize)
    _, X_test, _ = normalize_features(X_train, X_test, method=normalize)
    
    # Create dataset info
    info = DatasetInfo(
        name=name,
        num_features=X_train.shape[1],
        num_classes=len(label_encoder.classes_),
        class_names=list(label_encoder.classes_),
        train_samples=len(X_train),
        val_samples=len(X_val),
        test_samples=len(X_test),
    )
    
    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "X_val": X_val.astype(np.float32),
        "y_val": y_val,
        "X_test": X_test.astype(np.float32),
        "y_test": y_test,
        "info": info,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": selected_features,
    }
