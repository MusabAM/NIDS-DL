import json
import os

filepath = "notebooks/03_quantum_dl/03_vqc_cicids2018.ipynb"

if not os.path.exists(filepath):
    print("Notebook file not found.")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install PennyLane to run on Google Colab\n",
        "!pip install pennylane --quiet\n",
        "!pip install pennylane-lightning --quiet\n",
    ],
}

# Insert after the first markdown cell
nb["cells"].insert(1, new_cell)

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Added Colab install cell successfully.")
