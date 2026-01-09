"""
Download UNSW-NB15 Dataset - Alternative Sources
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination, description="Downloading"):
    """Download a file with progress bar."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, stream=True, timeout=120, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=description
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        
        # Verify file size is reasonable (> 1MB)
        if destination.stat().st_size < 1_000_000:
            print(f"  Warning: File seems too small, might be an error page")
            return False
            
        print(f"✓ Downloaded: {destination}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  UNSW-NB15 Dataset Downloader v2")
    print("="*60 + "\n")
    
    # Create directory
    data_dir = Path("./data/raw/unsw-nb15")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try multiple known sources
    sources = [
        # Source 1: CloudFlare CDN mirror commonly used
        {
            "name": "CloudFlare CDN",
            "training": "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_training-set.csv",
            "testing": "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_testing-set.csv",
        },
        # Source 2: Figshare
        {
            "name": "Figshare",
            "training": "https://figshare.com/ndownloader/files/24226893",
            "testing": "https://figshare.com/ndownloader/files/24226896",
        },
    ]
    
    files_to_download = {
        "UNSW_NB15_training-set.csv": "training",
        "UNSW_NB15_testing-set.csv": "testing",
    }
    
    success_count = 0
    
    for filename, key in files_to_download.items():
        dest = data_dir / filename
        
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"✓ {filename} already exists ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            success_count += 1
            continue
        
        print(f"\nDownloading {filename}...")
        
        downloaded = False
        for source in sources:
            print(f"  Trying {source['name']}...")
            url = source[key]
            if download_file(url, dest, filename):
                downloaded = True
                success_count += 1
                break
        
        if not downloaded:
            print(f"✗ Could not download {filename}")
    
    print("\n" + "="*60)
    
    if success_count == len(files_to_download):
        print(f"  ✓ Successfully prepared {success_count}/{len(files_to_download)} files!")
        print(f"  Location: {data_dir.absolute()}")
        
        # Show file info
        print("\n  Files:")
        total_size = 0
        for f in data_dir.glob("*.csv"):
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"    - {f.name}: {size_mb:.2f} MB")
        print(f"\n  Total: {total_size:.2f} MB")
    else:
        print(f"  Downloaded {success_count}/{len(files_to_download)} files")
        print("\n  [!] Manual download required:")
        print("  1. Open: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print("  2. Download and extract the CSV files")
        print(f"  3. Place them in: {data_dir.absolute()}")
        print("\n  Or use Google Drive link:")
        print("  https://drive.google.com/drive/folders/1tBQeXYLxqBR_6n8RQWzzXJcJQZWnzJAi")
    
    print("="*60 + "\n")
    
    return success_count == len(files_to_download)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
