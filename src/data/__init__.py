"""
Data module for NIDS-DL project.
Handles dataset downloading, preprocessing, and loading.
"""

from .download import DatasetDownloader
from .preprocessing import (
    preprocess_nsl_kdd,
    preprocess_cicids,
    preprocess_unsw_nb15,
    normalize_features,
    encode_labels,
    handle_class_imbalance,
)
from .datasets import (
    NIDSDataset,
    get_dataset,
    get_dataloaders,
)

__all__ = [
    "DatasetDownloader",
    "preprocess_nsl_kdd",
    "preprocess_cicids",
    "preprocess_unsw_nb15",
    "normalize_features",
    "encode_labels",
    "handle_class_imbalance",
    "NIDSDataset",
    "get_dataset",
    "get_dataloaders",
]
