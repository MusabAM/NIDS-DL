import json

notebook_path = (
    "c:/Users/musab/Projects/NIDS-DL/notebooks/02_classical_dl/04_cnn_cicids2018.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            new_line = line
            new_line = new_line.replace(
                "df.dropna(inplace=True)", "df.fillna(0, inplace=True)"
            )
            new_line = new_line.replace(
                "# 2. Replace Inf with NaN and drop NaNs",
                "# 2. Replace Inf with NaN and fill NaNs with 0",
            )
            new_line = new_line.replace(
                "Samples after dropping NaN/Inf", "Samples after filling NaN/Inf"
            )
            new_line = new_line.replace(
                "# Drop any new NaNs from conversion",
                "# Fill any new NaNs from conversion with 0",
            )
            new_source.append(new_line)
        cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4)

print("Modified notebook successfully.")
