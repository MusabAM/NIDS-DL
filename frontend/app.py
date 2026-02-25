import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from frontend import utils

# Page Config
st.set_page_config(
    page_title="NIDS-DL Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🛡️ NIDS-DL System")
page = st.sidebar.radio(
    "Navigation", ["Dashboard", "Live Prediction", "Batch Analysis"]
)

st.sidebar.markdown("---")
# Dataset Selection Dropdown
dataset_name = st.sidebar.selectbox(
    "Select Dataset", ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
)

# Model Selection Dropdown
# Filter available models based on dataset
available_models = list(utils.DATASET_CONFIG[dataset_name]["models"].keys())
model_type = st.sidebar.selectbox("Select Model", available_models)

st.sidebar.info(
    "**Status**: Online\n\n"
    f"**Dataset**: {dataset_name}\n\n"
    f"**Model**: {model_type}\n\n"
    "**Device**: " + ("CUDA 🟢" if torch.cuda.is_available() else "CPU 🟠")
)

# Load Resources
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_data
def get_cached_feature_columns(dataset_name):
    return utils.load_feature_columns(dataset_name)


try:
    feature_cols = get_cached_feature_columns(dataset_name)
    model, scaler, encoders = utils.load_model_and_scaler(
        model_type, dataset_name, device
    )
    model_loaded = model is not None
    if not model_loaded:
        st.sidebar.error(
            f"Failed to load {model_type} model. Check if file exists in results/models/."
        )
except Exception as e:
    st.error(f"Error loading resources: {e}")
    model_loaded = False

# ==============================================================================
# PAGE: Dashboard
# ==============================================================================
if page == "Dashboard":
    st.markdown(
        '<div class="main-header">Network Intrusion Detection System</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Deep Learning Optimized Security Monitoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="card">
            <h4>System Status</h4>
            <div class="metric-value">Active</div>
            <p>Monitoring enabled</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        acc = "88.78%"  # Placeholder or load from results
        st.markdown(
            f"""
        <div class="card">
            <h4>Model Accuracy</h4>
            <div class="metric-value">{acc}</div>
            <p>On UNSW-NB15 Test Set</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="card">
            <h4>Supported Attacks</h4>
            <div class="metric-value">39+</div>
            <p>DoS, Probe, R2L, U2R</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("### Model Performance Overview")

    # Placeholder data for visualization
    chart_data = pd.DataFrame(
        {
            "Model": ["CNN", "LSTM", "Transformer", "Autoencoder"],
            "Accuracy": [88.78, 88.48, 87.35, 88.04],
            "F1-Score": [0.89, 0.89, 0.88, 0.88],
        }
    )

    fig = px.bar(
        chart_data,
        x="Model",
        y=["Accuracy", "F1-Score"],
        barmode="group",
        title="Comparative Analysis of Deep Learning Models",
        color_discrete_sequence=["#1E88E5", "#FFC107"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Alerts")
    st.info("No recent high-severity alerts detected.")

# ==============================================================================
# PAGE: Live Prediction
# ==============================================================================
elif page == "Live Prediction":
    st.header("🔍 Live Traffic Analysis")

    if not model_loaded:
        st.warning("⚠️ Model not loaded. Please train the model or check paths.")
        st.stop()

    if dataset_name != "NSL-KDD":
        st.warning(
            f"⚠️ Live Prediction form is currently optimized for NSL-KDD. {dataset_name} has different features."
        )
        st.info(
            "Please use Batch Analysis for this dataset, or Switch to NSL-KDD for live form demo."
        )
        # We could implement dynamic forms here, but for now we restrict it.
        # Allowing it might cause shape mismatch errors if we don't handle 41 vs 42 vs 77 features.

        # Optional: Allow upload of single sample JSON/CSV for other datasets?
        # For now, just stop.
        # st.stop()
        # Actually, let's just show the form but warn that it might fail if features don't match.
        # But wait, the form has specific fields like 'flag' etc which might not exist or be different.
        # UNSW has 'state' instead of 'flag', 'proto' instead of 'protocol_type'.
        # CICIDS has completely different flow features.
        # So stopping is safer.
        st.stop()

    st.markdown("Enter network flow parameters to classify traffic.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input("Duration", min_value=0, value=0)
            src_bytes = st.number_input("Source Bytes", min_value=0, value=100)
            dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0)
            protocol = st.selectbox("Protocol", ["tcp", "udp", "icmp"])

        with col2:
            service = st.selectbox(
                "Service", ["http", "private", "ftp_data", "smtp", "other"]
            )
            flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR"])
            count = st.number_input("Count (Traffic Rate)", min_value=0, value=1)

        # Add hidden details expander for more features if needed
        with st.expander("Advanced Network Features"):
            serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0)
            rerror_rate = st.slider("REJ Error Rate", 0.0, 1.0, 0.0)
            same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0)

        submitted = st.form_submit_button("Analyze Traffic")

    if submitted:
        # Construct DataFrame
        # NOTE: This assumes we populate ONLY the critical fields used by the model
        # For a real implementation, we'd need to populate ALL 41 features with defaults

        input_data = {
            "duration": duration,
            "protocol_type": protocol,
            "service": service,
            "flag": flag,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "count": count,
            "serror_rate": serror_rate,
            "rerror_rate": rerror_rate,
            "same_srv_rate": same_srv_rate,
            # Defaults for others
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "srv_count": count,
            "srv_serror_rate": serror_rate,
            "srv_rerror_rate": rerror_rate,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 1,
            "dst_host_srv_count": 1,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": serror_rate,
            "dst_host_srv_serror_rate": serror_rate,
            "dst_host_rerror_rate": rerror_rate,
            "dst_host_srv_rerror_rate": rerror_rate,
        }

        df = pd.DataFrame([input_data])

        # Preprocess
        X_scaled = utils.preprocess_input(df, scaler, feature_cols)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # Predict
        with torch.no_grad():
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

        # Result
        st.markdown("### Analysis Result")

        if pred_class == 1:
            st.error(f"🚨 **Threat Detected: ATTACK**")
            label = (
                "Reconstruction Error" if model_type == "Autoencoder" else "Confidence"
            )
            val = (
                f"{confidence:.4f}"
                if model_type == "Autoencoder"
                else f"{confidence*100:.2f}%"
            )
            st.metric(label, val)
        else:
            st.success(f"✅ **Traffic Status: NORMAL**")
            label = (
                "Reconstruction Error" if model_type == "Autoencoder" else "Confidence"
            )
            val = (
                f"{confidence:.4f}"
                if model_type == "Autoencoder"
                else f"{confidence*100:.2f}%"
            )
            st.metric(label, val)

        # Feature contribution (simple bar chart of input values for visual)
        st.bar_chart(df[["src_bytes", "dst_bytes", "count"]].T)

# ==============================================================================
# PAGE: Batch Analysis
# ==============================================================================
elif page == "Batch Analysis":
    st.header("📂 Batch File Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV / PCAP (Pre-processed)", type=["csv", "txt"]
    )

    if uploaded_file and model_loaded:
        try:
            # Assuming NSL-KDD format (no header or specific header)
            # For simplicity, assuming user uploads a file with headers or we use COLUMNS
            # Tricky part: Uploaded file might not have headers.
            # We'll try to read it.

            df = pd.read_csv(uploaded_file)

            # If columns don't match, warn user
            # (Simplification: We assume valid schema for now)

            st.write(f"Loaded {len(df)} samples.")

            if st.button("Run Predictions"):
                progress = st.progress(0)

                # Preprocess
                # Handle missing columns if dataset is partial
                # For NSL-KDD we use header constants if missing
                if dataset_name == "NSL-KDD":
                    missing_cols = [
                        c for c in utils.COLUMNS[:-2] if c not in df.columns
                    ]
                else:
                    missing_cols = []  # We rely on feature_cols matching

                # If too many missing, maybe input format is raw?
                # If too many missing, maybe input format is raw?
                if len(missing_cols) > 20:
                    # Try setting header=None and names=COLUMNS
                    uploaded_file.seek(0)
                    if dataset_name == "NSL-KDD":
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file, header=None, names=utils.COLUMNS
                        )

                X_scaled = utils.preprocess_input(
                    df, scaler, feature_cols, encoders, dataset_name
                )

                # Batch Predict (in chunks to save memory if large)
                batch_size = 1000
                all_preds = []
                all_probs = []

                with torch.no_grad():
                    for i in range(0, len(X_scaled), batch_size):
                        X_batch = torch.FloatTensor(X_scaled[i : i + batch_size]).to(
                            device
                        )
                        if model_type == "Autoencoder":
                            losses = model.reconstruction_error(X_batch)
                            preds = (losses > 0.1).long()  # Threshold 0.1
                            all_preds.extend(preds.cpu().numpy())
                            all_probs.extend(losses.cpu().numpy())
                        else:
                            outputs = model(X_batch)
                            probs = torch.softmax(outputs, dim=1)
                            preds = torch.argmax(probs, dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_probs.extend(probs[:, 1].cpu().numpy())

                        progress.progress(min(1.0, (i + batch_size) / len(X_scaled)))

                df["Prediction"] = ["Attack" if p == 1 else "Normal" for p in all_preds]
                df["Attack_Probability"] = all_probs

                st.success("Analysis Complete!")

                # Statistics
                st.markdown("### Summary Statistics")
                col1, col2 = st.columns(2)
                n_attacks = sum(all_preds)
                n_normal = len(all_preds) - n_attacks

                col1.metric("Total Attacks Found", n_attacks)
                col2.metric("Normal Connections", n_normal)

                # Pie Chart
                counts = pd.DataFrame(
                    {"Type": ["Normal", "Attack"], "Count": [n_normal, n_attacks]}
                )
                fig = px.pie(
                    counts,
                    values="Count",
                    names="Type",
                    title="Traffic Distribution",
                    color="Type",
                    color_discrete_map={"Normal": "#66BB6A", "Attack": "#EF5350"},
                )
                st.plotly_chart(fig)

                # Data Table
                st.dataframe(df.head(100))

                # Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV", csv, "results.csv", "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
