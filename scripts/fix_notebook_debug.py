import json

filepath = "notebooks/03_quantum_dl/03_vqc_cicids2018.ipynb"
with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        if any("def clean_cicids2018(df):" in line for line in source):
            new_source = []
            for line in source:
                new_source.append(line)
                if "drop_cols if c in df_clean.columns" in line:
                    new_source.append(
                        '    print("Columns after drop:", df_clean.columns.tolist())\\n'
                    )
            cell["source"] = new_source

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Injected debug statement.")
