import json

filepath = "notebooks/03_quantum_dl/03_vqc_cicids2018.ipynb"
with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        if any("def clean_cicids2018(df):" in line for line in source):
            new_source = []
            i = 0
            while i < len(source):
                line = source[i]
                if "# Handle Label" in line:
                    new_source.extend(
                        [
                            "    # Handle Label (case-insensitive)\n",
                            "    label_col = None\n",
                            "    for col in df_clean.columns:\n",
                            '        if col.lower() == "label":\n',
                            "            label_col = col\n",
                            "            break\n",
                            "\n",
                            "    if label_col is not None:\n",
                            '        df_clean["binary_label"] = df_clean[label_col].apply(lambda x: 0 if x == "Benign" else 1)\n',
                            "        df_clean = df_clean.drop(label_col, axis=1)\n",
                            "    else:\n",
                            '        print("WARNING: Label column not found.")\n',
                        ]
                    )
                    # Skip the old code block
                    while i < len(source) - 1 and not source[i + 1].startswith(
                        "    # Handle Inf"
                    ):
                        i += 1
                else:
                    new_source.append(line)
                i += 1
            cell["source"] = new_source

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Updated the preprocessing cell in the ipynb correctly.")
