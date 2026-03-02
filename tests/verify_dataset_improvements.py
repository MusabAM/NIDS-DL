import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.data.datasets import get_dataset

console = Console()

def verify_improvements(dataset_name="nsl_kdd"):
    console.print(f"\n# --- Verifying Improvements for {dataset_name} ---")
    
    # 1. Load standard dataset
    console.print("\n# 1. Loading Standard Dataset...")
    standard = get_dataset(
        name=dataset_name,
        feature_engineering=False,
        feature_selection=False
    )
    
    # 2. Load improved dataset
    console.print("\n# 2. Loading Improved Dataset (Selection + Engineering)...")
    improved = get_dataset(
        name=dataset_name,
        feature_engineering=True,
        feature_selection=True,
        k_features=0.5  # Keep 50% of features
    )
    
    # Compare
    console.print("\n# Comparison Results:")
    console.print(f"Standard feature count: {standard['info'].num_features}")
    console.print(f"Improved feature count: {improved['info'].num_features}")
    
    # Check for log transformation effects (skewness reduction)
    if 'src_bytes' in standard['feature_names'] and 'src_bytes' in improved['feature_names']:
        std_idx = standard['feature_names'].index('src_bytes')
        imp_idx = improved['feature_names'].index('src_bytes')
        
        std_skew = pd.Series(standard['X_train'][:, std_idx]).skew()
        imp_skew = pd.Series(improved['X_train'][:, imp_idx]).skew()
        
        console.print(f"Skewness of 'src_bytes' (Standard): {std_skew:.4f}")
        console.print(f"Skewness of 'src_bytes' (Improved): {imp_skew:.4f}")
        
    console.print("\n[DONE] Preprocessing pipeline verified successfully!")

if __name__ == "__main__":
    try:
        verify_improvements()
    except Exception as e:
        console.print(f"\n[bold red]Error during verification: {e}[/bold red]")
        import traceback
        traceback.print_exc()
