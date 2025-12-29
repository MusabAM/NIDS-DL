#!/usr/bin/env python
"""
Script to verify the NIDS-DL environment setup.
Checks Python version, GPU availability, and required packages.
"""

import sys
import platform
from pathlib import Path


def print_header(msg: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def print_status(name: str, status: bool, info: str = ""):
    """Print status with checkmark or cross."""
    symbol = "✓" if status else "✗"
    status_str = f"[{symbol}] {name}"
    if info:
        status_str += f": {info}"
    print(status_str)


def check_python_version():
    """Check Python version."""
    print_header("Python Environment")
    
    version = platform.python_version()
    major, minor = sys.version_info[:2]
    
    # We want Python 3.11 or 3.12
    is_valid = (major == 3 and minor in [11, 12])
    
    print_status(
        "Python Version",
        is_valid,
        f"{version} ({'OK' if is_valid else 'Recommend 3.11 or 3.12'})"
    )
    print(f"    Path: {sys.executable}")
    
    return is_valid


def check_cuda():
    """Check CUDA availability."""
    print_header("GPU / CUDA")
    
    cuda_available = False
    cuda_version = "N/A"
    gpu_name = "N/A"
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print_status("PyTorch CUDA", cuda_available, f"CUDA {cuda_version}" if cuda_available else "Not available")
        if cuda_available:
            print(f"    GPU: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f} GB")
    except ImportError:
        print_status("PyTorch", False, "Not installed")
    
    try:
        import tensorflow as tf
        tf_gpus = tf.config.list_physical_devices('GPU')
        tf_cuda = len(tf_gpus) > 0
        print_status("TensorFlow GPU", tf_cuda, f"{len(tf_gpus)} GPU(s)" if tf_cuda else "Not available")
    except ImportError:
        print_status("TensorFlow", False, "Not installed")
    
    return cuda_available


def check_packages():
    """Check required packages."""
    print_header("Core Packages")
    
    packages = {
        # Core
        "numpy": "numpy",
        "pandas": "pandas", 
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        
        # Deep Learning
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        
        # Visualization
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "plotly": "plotly",
        
        # Utilities
        "tqdm": "tqdm",
        "rich": "rich",
        "yaml": "PyYAML",
    }
    
    all_installed = True
    for module, name in packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "unknown")
            print_status(name, True, version)
        except ImportError:
            print_status(name, False, "Not installed")
            all_installed = False
    
    return all_installed


def check_quantum_packages():
    """Check quantum packages."""
    print_header("Quantum Packages")
    
    packages = {
        "pennylane": "PennyLane",
        "cirq": "Cirq",
        "tensorflow_quantum": "TensorFlow Quantum",
    }
    
    for module, name in packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "unknown")
            print_status(name, True, version)
        except ImportError:
            print_status(name, False, "Not installed (optional)")


def check_project_structure():
    """Check project structure."""
    print_header("Project Structure")
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/data",
        "src/models/classical",
        "src/models/quantum",
        "src/training",
        "src/evaluation",
        "src/utils",
        "configs",
        "scripts",
        "results",
    ]
    
    project_root = Path(__file__).parent.parent
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print_status(dir_path, exists)
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """Main verification routine."""
    print("\n" + "=" * 60)
    print("  NIDS-DL Environment Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Python", check_python_version()))
    results.append(("CUDA", check_cuda()))
    results.append(("Core Packages", check_packages()))
    check_quantum_packages()
    results.append(("Project Structure", check_project_structure()))
    
    # Summary
    print_header("Summary")
    
    all_passed = True
    for name, passed in results:
        print_status(name, passed)
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed! Environment is ready.")
    else:
        print("\n⚠ Some checks failed. See above for details.")
        print("  Run: pip install -r requirements.txt")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
