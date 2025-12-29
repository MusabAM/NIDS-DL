"""
Logging utilities for NIDS-DL project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Rich console handler
    console_handler = RichHandler(
        console=Console(),
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%X]",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = "./results/logs",
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for an experiment.
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        
    Returns:
        Main logger
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}.log"
    
    # Configure root logger
    logger = get_logger("nids_dl", level=level, log_file=str(log_file))
    
    logger.info(f"Logging initialized: {log_file}")
    
    return logger


class TrainingLogger:
    """
    Logger for training metrics.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./results/logs",
    ):
        """
        Initialize training logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(
            f"training.{experiment_name}",
            log_file=str(self.log_dir / f"{experiment_name}_training.log")
        )
        
        self.epoch_metrics = []
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        **kwargs
    ):
        """Log metrics for an epoch."""
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            **kwargs
        }
        self.epoch_metrics.append(metrics)
        
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.4f}"
        if val_acc is not None:
            msg += f", val_acc={val_acc:.4f}"
        
        self.logger.info(msg)
    
    def log_test_results(
        self,
        test_loss: float,
        test_acc: float,
        **kwargs
    ):
        """Log test results."""
        self.logger.info(f"Test Results: loss={test_loss:.4f}, accuracy={test_acc:.4f}")
        for key, value in kwargs.items():
            self.logger.info(f"  {key}: {value}")
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save epoch metrics to file."""
        import json
        
        if filename is None:
            filename = self.log_dir / f"{self.experiment_name}_metrics.json"
        
        with open(filename, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filename}")
