"""
Configuration utilities for NIDS-DL project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_all_configs(configs_dir: str = "./configs") -> Dict[str, Any]:
    """
    Load all configuration files from directory.
    
    Args:
        configs_dir: Path to configs directory
        
    Returns:
        Dictionary with all configurations
    """
    configs = {}
    configs_path = Path(configs_dir)
    
    for config_file in configs_path.glob("*.yaml"):
        name = config_file.stem
        configs[name] = load_config(config_file)
    
    return configs


def get_device(device: str = "auto") -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def setup_gpu_memory_growth():
    """
    Configure TensorFlow to use GPU memory growth.
    Prevents TensorFlow from allocating all GPU memory at once.
    """
    if not TF_AVAILABLE:
        return
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth setup failed: {e}")


def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to {seed}")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_results_dir() -> Path:
    """Get the results directory."""
    return get_project_root() / "results"
