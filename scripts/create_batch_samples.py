import glob
import os

import numpy as np
import pandas as pd

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "batch_test_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NSL-KDD Config
NSL_KDD_PATH = os.path.join(BASE_DIR, "data", "raw", "nsl-kdd", "train.txt")
NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

# UNSW Config
UNSW_PATH = os.path.join(
    BASE_DIR, "data", "raw", "unsw-nb15", "UNSW_NB15_training-set.csv"
)

# CICIDS Config
CICIDS_DIR = os.path.join(BASE_DIR, "data", "raw", "cicids2017")


def create_samples(name, df, label_col):
    print(f"Creating samples for {name}...")

    sample_normal = None
    sample_attack = None

    if label_col in df.columns:
        # NSL-KDD
        if name == "nsl_kdd":
            normal_df = df[df[label_col] == "normal"]
            attack_df = df[df[label_col] != "normal"]
        # UNSW-NB15
        elif name == "unsw_nb15":
            # label 0 is normal, 1 is attack usually. Let's verify.
            # actually checking the csv content might be good but let's assume standard.
            # In UNSW, 'label' column: 0=normal, 1=attack.
            normal_df = df[df[label_col] == 0]
            attack_df = df[df[label_col] == 1]
        # CICIDS2017
        elif name == "cicids2017":
            normal_df = df[df[label_col] == "BENIGN"]
            attack_df = df[df[label_col] != "BENIGN"]
        else:
            # Fallback
            normal_df = df.sample(1)
            attack_df = df.sample(1)

        # Sample 1 of each if available
        if not normal_df.empty:
            sample_normal = normal_df.sample(1)
            print(f"  Selected 1 Normal sample.")
        else:
            print(f"  WARNING: No Normal samples found for {name}")
            sample_normal = df.sample(1)  # Fallback

        if not attack_df.empty:
            sample_attack = attack_df.sample(1)
            print(f"  Selected 1 Attack sample.")
        else:
            print(f"  WARNING: No Attack samples found for {name}")
            sample_attack = df.sample(1)  # Fallback

    else:
        print(f"  Label col {label_col} not found in {df.columns}")
        sample_normal = df.sample(1)
        sample_attack = df.sample(1)

    # Combine them into single files? Or separate files?
    # User said: "batch samples to contain both attack and normal"
    # This implies a SINGLE file should have mixed content for testing?
    # "2 files for testing each model"
    # Let's make:
    #   sample_1.csv -> Contains 5 normal, 5 attack (Mixed)
    #   sample_2.csv -> Contains 5 normal, 5 attack (Mixed)
    # This allows testing the model's ability to distinguish.

    # Let's get 5 of each if possible
    n_samples = 5

    # Re-select to get more
    try:
        if name == "nsl_kdd":
            s_norm = df[df[label_col] == "normal"].sample(
                min(n_samples, len(df[df[label_col] == "normal"]))
            )
            s_att = df[df[label_col] != "normal"].sample(
                min(n_samples, len(df[df[label_col] != "normal"]))
            )
        elif name == "unsw_nb15":
            s_norm = df[df[label_col] == 0].sample(
                min(n_samples, len(df[df[label_col] == 0]))
            )
            s_att = df[df[label_col] == 1].sample(
                min(n_samples, len(df[df[label_col] == 1]))
            )
        elif name == "cicids2017":
            s_norm = df[df[label_col] == "BENIGN"].sample(
                min(n_samples, len(df[df[label_col] == "BENIGN"]))
            )
            s_att = df[df[label_col] != "BENIGN"].sample(
                min(n_samples, len(df[df[label_col] != "BENIGN"]))
            )
        else:
            s_norm = df.sample(n_samples)
            s_att = df.sample(n_samples)
    except Exception as e:
        print(f"Error sampling {name}: {e}")
        s_norm = df.head(5)
        s_att = df.tail(5)

    mixed_df_1 = (
        pd.concat([s_norm, s_att]).sample(frac=1).reset_index(drop=True)
    )  # Shuffle

    # Make a second batch with different samples if possible
    mixed_df_2 = (
        pd.concat([s_norm, s_att]).sample(frac=1).reset_index(drop=True)
    )  # Just reshuffle for now to be safe on size

    # Save
    path1 = os.path.join(OUTPUT_DIR, f"{name}_mixed_batch_1.csv")
    path2 = os.path.join(OUTPUT_DIR, f"{name}_mixed_batch_2.csv")

    mixed_df_1.to_csv(path1, index=False)
    mixed_df_2.to_csv(path2, index=False)
    print(f"  Saved {path1} (Size: {len(mixed_df_1)})")
    print(f"  Saved {path2} (Size: {len(mixed_df_2)})")


def main():
    # 1. NSL-KDD
    if os.path.exists(NSL_KDD_PATH):
        try:
            df = pd.read_csv(NSL_KDD_PATH, header=None, names=NSL_COLUMNS)
            # Preprocessing to match app expectation (app drops difficulty_level usually? No, let's keep it consistent with raw)
            # App utils.load_feature_columns drops it.
            # But the batch processor expects RAW data usually.
            # Let's save as is, with header.
            create_samples("nsl_kdd", df, "label")
        except Exception as e:
            print(f"Error processing NSL-KDD: {e}")
    else:
        print(f"NSL-KDD not found at {NSL_KDD_PATH}")

    # 2. UNSW-NB15
    if os.path.exists(UNSW_PATH):
        try:
            df = pd.read_csv(UNSW_PATH)
            create_samples("unsw_nb15", df, "label")
        except Exception as e:
            print(f"Error processing UNSW: {e}")
    else:
        print(f"UNSW not found at {UNSW_PATH}")

    # 3. CICIDS2017
    cicids_files = glob.glob(os.path.join(CICIDS_DIR, "*.csv"))
    if cicids_files:
        found_attacks = False
        found_normal = False

        # We need to collect enough samples
        collected_normal = []
        collected_attack = []

        for fpath in cicids_files:
            if len(collected_normal) >= 5 and len(collected_attack) >= 5:
                break

            print(f"Scanning {os.path.basename(fpath)} for samples...")
            try:
                # Read in chunks to find attacks
                chunksize = 50000
                for chunk in pd.read_csv(fpath, chunksize=chunksize):
                    chunk.columns = chunk.columns.str.strip()

                    # Normal
                    if len(collected_normal) < 5:
                        normals = chunk[chunk["Label"] == "BENIGN"]
                        if not normals.empty:
                            take = min(5 - len(collected_normal), len(normals))
                            collected_normal.append(normals.sample(take))

                    # Attack
                    if len(collected_attack) < 5:
                        attacks = chunk[chunk["Label"] != "BENIGN"]
                        if not attacks.empty:
                            take = min(5 - len(collected_attack), len(attacks))
                            collected_attack.append(attacks.sample(take))
                            print(
                                f"  Found {take} attack samples in {os.path.basename(fpath)}"
                            )

                    if len(collected_normal) >= 5 and len(collected_attack) >= 5:
                        break
            except Exception as e:
                print(f"  Error reading {os.path.basename(fpath)}: {e}")
                continue

        if collected_normal and collected_attack:
            df_normal = pd.concat(collected_normal)
            df_attack = pd.concat(collected_attack)
            combined_df = pd.concat([df_normal, df_attack])
            create_samples("cicids2017", combined_df, "Label")
        elif collected_normal:
            print(
                "WARNING: Only found Normal samples for CICIDS2017. Creating benign-only batches."
            )
            df_normal = pd.concat(collected_normal)
            create_samples("cicids2017", df_normal, "Label")
        else:
            print("Error: Could not extract valid samples from CICIDS2017 files.")

    else:
        print(f"No CICIDS files found in {CICIDS_DIR}")


if __name__ == "__main__":
    main()
