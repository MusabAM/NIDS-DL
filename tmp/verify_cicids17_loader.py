import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Paths
DATA_DIR = Path('data/raw/cicids2017')
csv_files = glob.glob(str(DATA_DIR / "*.csv"))

def verify_loader(files):
    if not files:
        print("No CSV files found!")
        return
    
    file = files[0]
    print(f"Testing loader with: {file}")
    df = pd.read_csv(file, nrows=100)
    
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Process Labels
    if 'Label' in df.columns:
        print(f"Original unique labels: {df['Label'].unique()}")
        df['binary_label'] = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
        df = df.drop('Label', axis=1)
    
    # Drop metadata
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Fwd Header Length.1']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Handle Numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Shape after cleaning: {df.shape}")
    if df.shape[0] > 0:
        print(f"Sample binary labels after cleaning: {df['binary_label'].unique()}")
    else:
        print("Warning: All rows were dropped! Investigating...")
        # Check instance of one row
        df_raw = pd.read_csv(file, nrows=1)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        for col in df_raw.columns:
            val = df_raw[col].iloc[0]
            try:
                pd.to_numeric(val)
            except:
                print(f"Non-numeric column found: {col} data: {val}")

if __name__ == "__main__":
    verify_loader(csv_files)
