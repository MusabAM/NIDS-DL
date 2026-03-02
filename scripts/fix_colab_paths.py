import json
import os

filepath = "notebooks/03_quantum_dl/03_vqc_cicids2018.ipynb"

if not os.path.exists(filepath):
    print("Notebook file not found.")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell["source"]
        if len(source) > 0 and source[0].startswith("# --- Data Paths ---"):
            new_source = [
                "# --- Data Paths ---\n",
                "try:\n",
                "    import google.colab\n",
                "    IN_COLAB = True\n",
                "except ImportError:\n",
                "    IN_COLAB = False\n",
                "\n",
                "if IN_COLAB:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    # IMPORTANT: Update this path if you uploaded NIDS-DL to a different folder in Google Drive\n",
                "    DATA_DIR = Path('/content/drive/MyDrive/NIDS-DL/data/raw/cicids2018')\n",
                "    RESULTS_DIR = Path('/content/drive/MyDrive/NIDS-DL/results/models/quantum')\n",
                "else:\n",
                "    DATA_DIR = project_root / 'data' / 'raw' / 'cicids2018'\n",
                "    RESULTS_DIR = project_root / 'results' / 'models' / 'quantum'\n",
                "\n",
                "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
                "\n",
            ]

            start_append = False
            for line in source:
                if "# --- Quantum Parameters ---" in line:
                    start_append = True
                if start_append:
                    new_source.append(line)

            cell["source"] = new_source

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Colab data paths successfully.")
