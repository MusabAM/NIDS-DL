import re


def resolve_app_py():
    with open(
        "c:/Users/musab/Projects/NIDS-DL/frontend/app.py", "r", encoding="utf-8"
    ) as f:
        content = f.read()

    # Chunk 1
    content = re.sub(
        r'<<<<<<< HEAD\n# Dataset Selection Dropdown\ndataset_name = st.sidebar.selectbox\(\n    "Select Dataset", \["NSL-KDD", "UNSW-NB15", "CICIDS2017"\]\n\)\n=======\n\n# Dataset Selection\ndataset = st\.sidebar\.selectbox\("Select Dataset", \["NSL-KDD", "UNSW-NB15", "CICIDS2017", "CICIDS2018"\]\)\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f',
        '# Dataset Selection Dropdown\ndataset_name = st.sidebar.selectbox(\n    "Select Dataset", ["NSL-KDD", "UNSW-NB15", "CICIDS2017", "CICIDS2018"]\n)',
        content,
    )

    # Chunk 2
    content = re.sub(
        r'<<<<<<< HEAD\n    f"\*\*Dataset\*\*: \{dataset_name\}\\n\\n"\n=======\n    f"\*\*Dataset\*\*: \{dataset\}\\n\\n"\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f',
        '    f"**Dataset**: {dataset_name}\\n\\n"',
        content,
    )

    # Replace occurrences of 'dataset' variable usages from INCOMING block inside app.py just in case
    # Let's fix Chunk 3
    content = re.sub(
        r"<<<<<<< HEAD\ndef get_cached_feature_columns\(dataset_name\):\n    return utils\.load_feature_columns\(dataset_name\)\n\n\ntry:\n    feature_cols = get_cached_feature_columns\(dataset_name\)\n    model, scaler, encoders = utils\.load_model_and_scaler\(\n        model_type, dataset_name, device\n    \)\n=======\ndef get_cached_feature_columns\(_dataset\):\n    return utils\.load_feature_columns\(_dataset\)\n\n\ntry:\n    feature_cols = get_cached_feature_columns\(dataset\)\n    # New signature: load_model_and_scaler\(model_name, dataset_name, device\)\n    model, scaler, encoders = utils\.load_model_and_scaler\(model_type, dataset, device\)\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f",
        "def get_cached_feature_columns(_dataset):\n    return utils.load_feature_columns(_dataset)\n\n\ntry:\n    feature_cols = get_cached_feature_columns(dataset_name)\n    model, scaler, encoders = utils.load_model_and_scaler(\n        model_type, dataset_name, device\n    )",
        content,
    )

    # Dataset check on dash
    content = re.sub(
        r'if dataset == "CICIDS2018":', 'if dataset_name == "CICIDS2018":', content
    )
    content = re.sub(
        r'f"Comparative Analysis of Deep Learning Models \(\{dataset\}\)"',
        'f"Comparative Analysis of Deep Learning Models ({dataset_name})"',
        content,
    )
    content = re.sub(
        r'f"Failed to load \{model_type\} model for \{dataset\}\. "',
        'f"Failed to load {model_type} model for {dataset_name}. "',
        content,
    )

    # Chunk 4
    content = re.sub(
        r'<<<<<<< HEAD(?:.|\n)*?# So stopping is safer\.\n        st\.stop\(\)\n\n    st\.markdown\("Enter network flow parameters to classify traffic\."\)\n=======\n    st\.markdown\(f"Enter network flow parameters to classify traffic\. \(\*\*\{dataset\}\*\* mode\)"\)\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f',
        '    if dataset_name not in ["NSL-KDD", "CICIDS2018"]:\n        st.warning(f"⚠️ Live Prediction form is not yet implemented for {dataset_name}.")\n        st.info("Please use Batch Analysis for this dataset.")\n        st.stop()\n\n    st.markdown(f"Enter network flow parameters to classify traffic. (**{dataset_name}** mode)")',
        content,
    )

    # Fix `if dataset == "NSL-KDD":`
    content = re.sub(
        r'if dataset == "NSL-KDD":\n        # NSL-KDD Live Prediction Form',
        'if dataset_name == "NSL-KDD":\n        # NSL-KDD Live Prediction Form',
        content,
    )
    # Fix `utils.preprocess_input(df, scaler, feature_cols, None, dataset)`
    content = re.sub(
        r"utils.preprocess_input\(df, scaler, feature_cols, None, dataset\)",
        "utils.preprocess_input(df, scaler, feature_cols, None, dataset_name)",
        content,
    )

    # Chunk 5 & 6 (The tricky one)
    # I will replace the entire block from `with torch.no_grad():` to the end of chunk 6 marker.
    to_replace = r'            with torch\.no_grad\(\):\n                outputs = model\(X_tensor\)(?:.|\n)*?<<<<<<< HEAD(?:.|\n)*?=======\n    elif dataset == "CICIDS2018":\n        # CICIDS2018 Live Prediction Form\n        with st\.form\("cicids_prediction_form"\):\n            st\.markdown\("#### CICIDS2018 Network Flow Features"\)\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f'

    replacement = """            with torch.no_grad():
                if model_type == "Autoencoder":
                    loss = model.reconstruction_error(X_tensor).item()
                    confidence = loss  # Use reconstruction error as score
                    threshold = 0.1  # Default threshold
                    pred_class = 1 if loss > threshold else 0
                else:
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_class].item()

            st.markdown("### Analysis Result")

            if pred_class == 1:
                st.error(f"🚨 **Threat Detected: ATTACK**")
                label = "Reconstruction Error" if model_type == "Autoencoder" else "Confidence"
                val = f"{confidence:.4f}" if model_type == "Autoencoder" else f"{confidence*100:.2f}%"
                st.metric(label, val)
            else:
                st.success(f"✅ **Traffic Status: NORMAL**")
                label = "Reconstruction Error" if model_type == "Autoencoder" else "Confidence"
                val = f"{confidence:.4f}" if model_type == "Autoencoder" else f"{confidence*100:.2f}%"
                st.metric(label, val)

            st.bar_chart(df[["src_bytes", "dst_bytes", "count"]].T)

    elif dataset_name == "CICIDS2018":
        # CICIDS2018 Live Prediction Form
        with st.form("cicids_prediction_form"):
            st.markdown("#### CICIDS2018 Network Flow Features")"""

    content = re.sub(to_replace, replacement, content)

    # Chunk 7 (Batch Analysis block)
    content = re.sub(
        r'<<<<<<< HEAD\n                # Preprocess(?:.|\n)*?=======\n                if dataset == "NSL-KDD":\n                    # NSL-KDD preprocessing\n                    missing_cols = \[\n                        c for c in utils\.NSL_KDD_COLUMNS\[:-2\] if c not in df\.columns\n                    \]\n                    if len\(missing_cols\) > 20:\n                        uploaded_file\.seek\(0\)\n                        df = pd\.read_csv\(\n                            uploaded_file,\n                            header=None,\n                            names=utils\.NSL_KDD_COLUMNS,\n                        \)\n\n                    X_scaled = utils\.preprocess_input\(\n                        df, scaler, feature_cols, None, dataset\n                    \)\n>>>>>>> 313b4ac6d9474d1e27a1b615d16c0a48f00d8a8f',
        """                # Preprocess
                if dataset_name == "NSL-KDD":
                    missing_cols = [c for c in utils.NSL_KDD_COLUMNS[:-2] if c not in df.columns]
                    if len(missing_cols) > 20:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, header=None, names=utils.NSL_KDD_COLUMNS)

                X_scaled = utils.preprocess_input(df, scaler, feature_cols, encoders, dataset_name)""",
        content,
    )

    # Fix any remaining dataset
    content = re.sub(
        r'st.markdown\(f"\*\*Dataset mode:\*\* \{dataset\}"\)',
        'st.markdown(f"**Dataset mode:** {dataset_name}")',
        content,
    )

    with open(
        "c:/Users/musab/Projects/NIDS-DL/frontend/app.py", "w", encoding="utf-8"
    ) as f:
        f.write(content)


resolve_app_py()
