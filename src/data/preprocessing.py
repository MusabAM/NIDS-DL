"""
Data preprocessing utilities for NIDS datasets.
Handles feature engineering, normalization, encoding, and class imbalance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from rich.console import Console

console = Console()


# ==============================================================================
# NSL-KDD Preprocessing
# ==============================================================================

NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# Attack type mapping for NSL-KDD
NSL_KDD_ATTACK_MAPPING = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
}


def preprocess_nsl_kdd(
    data_path: Union[str, Path],
    classification: str = "binary",
    include_difficulty: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess NSL-KDD dataset.
    
    Args:
        data_path: Path to NSL-KDD data file
        classification: 'binary' or 'multiclass'
        include_difficulty: Whether to include difficulty level as feature
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    console.print(f"[blue]Loading NSL-KDD from {data_path}[/blue]")
    
    # Load data
    df = pd.read_csv(data_path, header=None, names=NSL_KDD_COLUMNS)
    
    # Extract labels
    labels = df['label'].copy()
    
    # Map to attack categories for multiclass
    if classification == "multiclass":
        labels = labels.map(lambda x: NSL_KDD_ATTACK_MAPPING.get(x.lower(), 'Unknown'))
    else:  # binary
        labels = labels.apply(lambda x: 'Normal' if x.lower() == 'normal' else 'Attack')
    
    # Remove label and optionally difficulty level
    drop_cols = ['label']
    if not include_difficulty:
        drop_cols.append('difficulty_level')
    
    features = df.drop(columns=drop_cols)
    
    console.print(f"[green]✓[/green] Loaded {len(df)} samples, {len(features.columns)} features")
    console.print(f"[green]✓[/green] Label distribution:\n{labels.value_counts()}")
    
    return features, labels


# ==============================================================================
# CICIDS Preprocessing
# ==============================================================================

def preprocess_cicids(
    data_path: Union[str, Path],
    classification: str = "binary",
    drop_duplicates: bool = True,
    drop_nan: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess CICIDS2017/2018 dataset.
    
    Args:
        data_path: Path to CICIDS CSV file or directory
        classification: 'binary' or 'multiclass'
        drop_duplicates: Whether to drop duplicate rows
        drop_nan: Whether to drop rows with NaN values
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    console.print(f"[blue]Loading CICIDS from {data_path}[/blue]")
    
    path = Path(data_path)
    
    # Load single file or multiple files
    if path.is_file():
        df = pd.read_csv(path, low_memory=False)
    else:
        # Load all CSV files in directory
        csv_files = list(path.glob("*.csv"))
        console.print(f"[blue]Found {len(csv_files)} CSV files[/blue]")
        dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    
    # Clean column names (remove spaces, special chars)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    # Find label column (varies between versions)
    label_col = None
    for col in ['Label', 'label', 'Attack', 'attack']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in CICIDS data")
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop duplicates and NaN
    if drop_duplicates:
        initial_len = len(df)
        df = df.drop_duplicates()
        console.print(f"[yellow]Dropped {initial_len - len(df)} duplicate rows[/yellow]")
    
    if drop_nan:
        initial_len = len(df)
        df = df.dropna()
        console.print(f"[yellow]Dropped {initial_len - len(df)} rows with NaN[/yellow]")
    
    # Extract labels
    labels = df[label_col].copy()
    
    # Binary classification
    if classification == "binary":
        labels = labels.apply(lambda x: 'Normal' if x.upper() == 'BENIGN' else 'Attack')
    
    # Remove label column and any identifier columns
    drop_cols = [label_col]
    for col in ['Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp']:
        if col in df.columns:
            drop_cols.append(col)
    
    features = df.drop(columns=drop_cols)
    
    console.print(f"[green]✓[/green] Loaded {len(df)} samples, {len(features.columns)} features")
    console.print(f"[green]✓[/green] Label distribution:\n{labels.value_counts()}")
    
    return features, labels


# ==============================================================================
# UNSW-NB15 Preprocessing
# ==============================================================================

def preprocess_unsw_nb15(
    data_path: Union[str, Path],
    classification: str = "binary",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess UNSW-NB15 dataset.
    
    Args:
        data_path: Path to UNSW-NB15 CSV file or directory
        classification: 'binary' or 'multiclass'
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    console.print(f"[blue]Loading UNSW-NB15 from {data_path}[/blue]")
    
    path = Path(data_path)
    
    if path.is_file():
        df = pd.read_csv(path, low_memory=False)
    else:
        csv_files = list(path.glob("*.csv"))
        dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Handle labels
    if classification == "binary":
        # Use 'label' column (0=Normal, 1=Attack)
        if 'label' in df.columns:
            labels = df['label'].apply(lambda x: 'Normal' if x == 0 else 'Attack')
        else:
            labels = df['attack_cat'].apply(lambda x: 'Normal' if pd.isna(x) or x == '' else 'Attack')
    else:
        # Use 'attack_cat' for multiclass
        labels = df['attack_cat'].fillna('Normal')
    
    # Drop label columns and identifiers
    drop_cols = ['label', 'attack_cat', 'id', 'srcip', 'dstip', 'sport', 'dsport']
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    features = df.drop(columns=drop_cols)
    
    console.print(f"[green]✓[/green] Loaded {len(df)} samples, {len(features.columns)} features")
    console.print(f"[green]✓[/green] Label distribution:\n{labels.value_counts()}")
    
    return features, labels


# ==============================================================================
# Feature Engineering & Normalization
# ==============================================================================

def encode_categorical(
    df: pd.DataFrame,
    method: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features.
    
    Args:
        df: DataFrame with features
        method: 'label' or 'onehot'
        
    Returns:
        Tuple of (encoded DataFrame, encoders dict)
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    if not categorical_cols:
        return df, encoders
    
    console.print(f"[blue]Encoding {len(categorical_cols)} categorical columns[/blue]")
    
    if method == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    elif method == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df, encoders


def feature_engineering(
    df: pd.DataFrame,
    skew_threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Apply feature engineering like log transformation for skewed features.
    
    Args:
        df: Input DataFrame
        skew_threshold: Threshold for skewness to apply log transform
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Log transform highly skewed features
    skewed_features = df[numeric_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    high_skew = skewed_features[abs(skewed_features) > skew_threshold].index
    
    if len(high_skew) > 0:
        console.print(f"[blue]Applying log transformation to {len(high_skew)} skewed features[/blue]")
        for col in high_skew:
            # Add small constant to handle zeros
            df[col] = np.log1p(df[col] - df[col].min())
            
    return df


def select_features(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    k: Union[int, float] = 0.8,
    method: str = "mutual_info",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top features using statistical methods.
    
    Args:
        X: Feature DataFrame
        y: Labels
        k: Number of features (int) or proportion (float)
        method: 'mutual_info' or 'correlation'
        
    Returns:
        Tuple of (selected DataFrame, list of selected features)
    """
    if isinstance(k, float):
        k = int(len(X.columns) * k)
        
    console.print(f"[blue]Selecting top {k} features using {method}...[/blue]")
    
    if method == "mutual_info":
        if len(X) > 50000:
            X_sample = X.sample(50000, random_state=42)
            y_sample = y[:50000] if isinstance(y, np.ndarray) else y.iloc[:50000]
            selector = SelectKBest(mutual_info_classif, k=k)
            selector.fit(X_sample, y_sample)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
            selector.fit(X, y)
            
        selected_features = X.columns[selector.get_support()].tolist()
        
    elif method == "correlation":
        corr_matrix = X.corr().abs()
        upper = upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        selected_features = [c for c in X.columns if c not in to_drop]
        
        if len(selected_features) > k:
            selected_features = selected_features[:k]
            
    else:
        raise ValueError(f"Unknown selection method: {method}")
        
    console.print(f"[green]✓[/green] Selected {len(selected_features)} features")
    return X[selected_features], selected_features


def normalize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = "standard",
) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
    """
    Normalize features using specified method.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        method: 'standard', 'minmax', or 'robust'
        
    Returns:
        Tuple of (normalized X_train, normalized X_test, scaler)
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    
    if method not in scalers:
        raise ValueError(f"Unknown normalization method: {method}")
    
    scaler = scalers[method]
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    console.print(f"[green]✓[/green] Applied {method} normalization")
    
    return X_train_scaled, X_test_scaled, scaler


def encode_labels(
    y: pd.Series,
    encoding: str = "label",
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode labels.
    
    Args:
        y: Label Series
        encoding: 'label' for integer encoding
        
    Returns:
        Tuple of (encoded labels, encoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    console.print(f"[green]✓[/green] Label classes: {le.classes_}")
    
    return y_encoded, le


# ==============================================================================
# Class Imbalance Handling
# ==============================================================================

def handle_class_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance using resampling techniques.
    
    Args:
        X: Features
        y: Labels
        method: 'smote', 'adasyn', 'undersample', or 'none'
        random_state: Random seed
        
    Returns:
        Tuple of (resampled X, resampled y)
    """
    if method == "none":
        return X, y
    
    console.print(f"[blue]Applying {method} resampling...[/blue]")
    console.print(f"[blue]Original class distribution: {np.bincount(y)}[/blue]")
    
    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "adasyn":
        sampler = ADASYN(random_state=random_state)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    console.print(f"[green]✓[/green] Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled


# ==============================================================================
# Data Splitting
# ==============================================================================

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        stratify: Whether to stratify splits
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    stratify_y = y if stratify else None
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_y,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=stratify_temp,
        random_state=random_state
    )
    
    console.print(f"[green]✓[/green] Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
