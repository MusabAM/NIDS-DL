"""
Training utilities for NIDS-DL models.
Supports both PyTorch and TensorFlow/Keras training loops.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, List, Tuple
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm
from rich.console import Console

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..utils.logger import TrainingLogger

console = Console()


class Trainer(ABC):
    """Abstract base class for training."""
    
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str):
        pass


# ==============================================================================
# PyTorch Trainer
# ==============================================================================

class TorchTrainer(Trainer):
    """
    PyTorch model trainer with full training loop.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        experiment_name: str = "experiment",
        log_dir: str = "./results/logs",
    ):
        """
        Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Optional learning rate scheduler
            experiment_name: Name for logging
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        
        self.logger = TrainingLogger(experiment_name, log_dir)
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
                pbar.set_postfix(loss=loss.item(), acc=correct/total)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        model_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            model_path: Path to save best model
            
        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        
        patience_counter = 0
        
        console.print(f"\n[bold blue]Training for {epochs} epochs[/bold blue]\n")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            self.logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
            
            console.print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if save_best and model_path:
                    self.save(model_path)
                    console.print(f"[green]✓ New best model saved![/green]")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                console.print(f"\n[yellow]Early stopping at epoch {epoch}[/yellow]")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.logger.save_metrics()
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with test metrics
        """
        test_loss, test_acc = self.validate(test_loader)
        
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        
        self.logger.log_test_results(test_loss, test_acc)
        
        return results
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)


# ==============================================================================
# Keras Trainer
# ==============================================================================

class KerasTrainer(Trainer):
    """
    TensorFlow/Keras model trainer.
    """
    
    def __init__(
        self,
        model: "keras.Model",
        experiment_name: str = "experiment",
        log_dir: str = "./results/logs",
    ):
        """
        Initialize Keras trainer.
        
        Args:
            model: Compiled Keras model
            experiment_name: Name for logging
            log_dir: Directory for logs
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for KerasTrainer")
        
        self.model = model
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        model_path: Optional[str] = None,
    ) -> "keras.callbacks.History":
        """
        Train the Keras model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            model_path: Path to save best model
            
        Returns:
            Keras History object
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir / "tensorboard" / self.experiment_name),
            ),
        ]
        
        if save_best and model_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1,
                )
            )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        
        return history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary with test metrics
        """
        results = self.model.evaluate(X_test, y_test, return_dict=True)
        return results
    
    def save(self, path: str):
        """Save model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model."""
        self.model = keras.models.load_model(path)
