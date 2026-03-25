import argparse
import glob
import os

import pandas as pd

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw", "cicids2018")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "batch_test_samples")


def create_batches(num_batches=2, batch_size=10, normal_ratio=0.5):
    """
    Generates CSV batches for testing against models.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}.")
        return

    n_normal = int(batch_size * normal_ratio)
    n_attack = batch_size - n_normal

    for batch_idx in range(1, num_batches + 1):
        collected_normal = pd.DataFrame()
        collected_attack = pd.DataFrame()

        print(f"Generating Batch {batch_idx}...")

        for file in csv_files:
            if len(collected_normal) >= n_normal and len(collected_attack) >= n_attack:
                break

            try:
                # Read in chunks to avoid memory issues with large CICIDS2018 files
                for chunk in pd.read_csv(file, chunksize=50000, low_memory=False):
                    chunk.columns = chunk.columns.str.strip()

                    # Assume label column is 'Label'
                    if "Label" not in chunk.columns:
                        continue

                    # Clean label strings
                    chunk["Label"] = chunk["Label"].astype(str).str.strip()

                    # Normal traffic
                    if len(collected_normal) < n_normal:
                        normals = chunk[chunk["Label"].str.lower() == "benign"]
                        if not normals.empty:
                            take = min(n_normal - len(collected_normal), len(normals))
                            collected_normal = pd.concat(
                                [collected_normal, normals.sample(take)]
                            )

                    # Attack traffic
                    if len(collected_attack) < n_attack:
                        attacks = chunk[chunk["Label"].str.lower() != "benign"]
                        if not attacks.empty:
                            take = min(n_attack - len(collected_attack), len(attacks))
                            collected_attack = pd.concat(
                                [collected_attack, attacks.sample(take)]
                            )

                    if (
                        len(collected_normal) >= n_normal
                        and len(collected_attack) >= n_attack
                    ):
                        break
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

        # Combine and shuffle
        if not collected_normal.empty or not collected_attack.empty:
            batch_df = (
                pd.concat([collected_normal, collected_attack])
                .sample(frac=1)
                .reset_index(drop=True)
            )
            output_file = os.path.join(OUTPUT_DIR, f"cicids2018_batch_{batch_idx}.csv")
            batch_df.to_csv(output_file, index=False)
            print(
                f"  Created {output_file} (Total: {len(batch_df)}, Normal: {len(collected_normal)}, Attack: {len(collected_attack)})"
            )
        else:
            print("  Could not collect enough samples from the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CICIDS2018 test batches.")
    parser.add_argument(
        "--num_batches", type=int, default=2, help="Number of CSV batches to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Number of rows per batch"
    )
    parser.add_argument(
        "--normal_ratio",
        type=float,
        default=0.5,
        help="Ratio of normal traffic (0.0 to 1.0)",
    )
    args = parser.parse_args()

    create_batches(args.num_batches, args.batch_size, args.normal_ratio)
