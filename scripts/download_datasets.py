#!/usr/bin/env python
"""
Script to download NIDS datasets.
Usage: python download_datasets.py --dataset [nsl_kdd|cicids2017|cicids2018|unsw_nb15|all]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import DatasetDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download NIDS datasets for research"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nsl_kdd",
        choices=["nsl_kdd", "cicids2017", "cicids2018", "unsw_nb15", "all"],
        help="Dataset to download (default: nsl_kdd)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="Directory to save datasets (default: ./data/raw)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing datasets without downloading"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.verify:
        print("\nVerifying datasets...")
        datasets = ["nsl_kdd", "cicids2017", "cicids2018", "unsw_nb15"]
        for ds in datasets:
            downloader.verify_dataset(ds)
    else:
        downloader.download(args.dataset)


if __name__ == "__main__":
    main()
