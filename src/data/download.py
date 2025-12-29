"""
Dataset download utilities for NIDS-DL project.
Supports: NSL-KDD, CICIDS2017, CICIDS2018, UNSW-NB15
"""

import os
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, List
import requests
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


class DatasetDownloader:
    """
    Download and extract NIDS datasets.
    
    Supported datasets:
        - NSL-KDD: Classic benchmark dataset
        - CICIDS2017: Canadian Institute for Cybersecurity IDS 2017
        - CICIDS2018: Extended version with AWS infrastructure
        - UNSW-NB15: Australian network intrusion dataset
    """
    
    # Dataset download URLs
    DATASET_URLS = {
        "nsl_kdd": {
            "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
            "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
            "train_full": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
            "test_21": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest-21.txt",
        },
        # Note: CICIDS and UNSW-NB15 are large datasets that require manual download
        # from their official sources due to licensing and size constraints
    }
    
    # Column names for NSL-KDD
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
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir: Directory to save downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(
        self, 
        url: str, 
        destination: Path, 
        description: str = "Downloading"
    ) -> bool:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            destination: Path to save the file
            description: Description for progress bar
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True)
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
            
            console.print(f"[green]✓[/green] Downloaded: {destination.name}")
            return True
            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]✗[/red] Failed to download {url}: {e}")
            return False
    
    def download_nsl_kdd(self) -> bool:
        """
        Download NSL-KDD dataset.
        
        Returns:
            True if all files downloaded successfully
        """
        console.print("\n[bold blue]Downloading NSL-KDD Dataset[/bold blue]")
        
        dataset_dir = self.data_dir / "nsl-kdd"
        dataset_dir.mkdir(exist_ok=True)
        
        success = True
        for name, url in self.DATASET_URLS["nsl_kdd"].items():
            dest = dataset_dir / f"{name}.txt"
            if dest.exists():
                console.print(f"[yellow]![/yellow] {name}.txt already exists, skipping...")
                continue
            
            if not self.download_file(url, dest, f"NSL-KDD {name}"):
                success = False
        
        # Create column names file
        columns_file = dataset_dir / "columns.txt"
        with open(columns_file, 'w') as f:
            f.write('\n'.join(self.NSL_KDD_COLUMNS))
        
        return success
    
    def download_cicids2017(self) -> None:
        """
        Provide instructions for downloading CICIDS2017.
        
        Note: CICIDS2017 is ~6GB and requires download from the official UNB website.
        """
        console.print("\n[bold blue]CICIDS2017 Dataset Instructions[/bold blue]")
        console.print("""
[yellow]CICIDS2017 requires manual download due to size (~6GB).[/yellow]

Steps:
1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the CSV files (MachineLearningCVE folder)
3. Extract to: {data_dir}/cicids2017/

Files you'll get:
- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

Alternative: Use Kaggle (smaller preprocessed version)
- https://www.kaggle.com/datasets/cicdataset/cicids2017
        """.format(data_dir=self.data_dir))
    
    def download_cicids2018(self) -> None:
        """
        Provide instructions for downloading CICIDS2018.
        """
        console.print("\n[bold blue]CICIDS2018 Dataset Instructions[/bold blue]")
        console.print("""
[yellow]CICIDS2018 requires manual download from AWS (~16GB).[/yellow]

Steps:
1. Visit: https://www.unb.ca/cic/datasets/ids-2018.html
2. The dataset is hosted on AWS - follow the S3 bucket instructions
3. Extract to: {data_dir}/cicids2018/

Alternative: Use preprocessed versions from Kaggle
        """.format(data_dir=self.data_dir))
    
    def download_unsw_nb15(self) -> None:
        """
        Provide instructions for downloading UNSW-NB15.
        """
        console.print("\n[bold blue]UNSW-NB15 Dataset Instructions[/bold blue]")
        console.print("""
[yellow]UNSW-NB15 requires registration for download.[/yellow]

Steps:
1. Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Register and download the CSV files
3. Extract to: {data_dir}/unsw-nb15/

Files to download:
- UNSW-NB15_1.csv through UNSW-NB15_4.csv (raw data)
- UNSW_NB15_training-set.csv
- UNSW_NB15_testing-set.csv

Alternative: Kaggle
- https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
        """.format(data_dir=self.data_dir))
    
    def download(self, dataset: str = "all") -> None:
        """
        Download specified dataset(s).
        
        Args:
            dataset: Dataset name or 'all' for all datasets
        """
        datasets = ["nsl_kdd", "cicids2017", "cicids2018", "unsw_nb15"]
        
        if dataset == "all":
            for ds in datasets:
                self.download(ds)
        elif dataset == "nsl_kdd":
            self.download_nsl_kdd()
        elif dataset == "cicids2017":
            self.download_cicids2017()
        elif dataset == "cicids2018":
            self.download_cicids2018()
        elif dataset == "unsw_nb15":
            self.download_unsw_nb15()
        else:
            console.print(f"[red]Unknown dataset: {dataset}[/red]")
            console.print(f"Available: {', '.join(datasets)}")
    
    def verify_dataset(self, dataset: str) -> bool:
        """
        Verify that a dataset exists and has expected files.
        
        Args:
            dataset: Dataset name to verify
            
        Returns:
            True if dataset is present and valid
        """
        dataset_paths = {
            "nsl_kdd": self.data_dir / "nsl-kdd",
            "cicids2017": self.data_dir / "cicids2017",
            "cicids2018": self.data_dir / "cicids2018",
            "unsw_nb15": self.data_dir / "unsw-nb15",
        }
        
        if dataset not in dataset_paths:
            console.print(f"[red]Unknown dataset: {dataset}[/red]")
            return False
        
        path = dataset_paths[dataset]
        if not path.exists():
            console.print(f"[red]Dataset not found: {path}[/red]")
            return False
        
        files = list(path.glob("*.csv")) + list(path.glob("*.txt"))
        if not files:
            console.print(f"[yellow]No data files found in {path}[/yellow]")
            return False
        
        console.print(f"[green]✓[/green] {dataset}: {len(files)} files found")
        return True


if __name__ == "__main__":
    # Example usage
    downloader = DatasetDownloader()
    downloader.download("nsl_kdd")
