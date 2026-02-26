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

# Dataset Selection
dataset = st.sidebar.selectbox("Select Dataset", ["NSL-KDD", "UNSW-NB15", "CICIDS2017", "CICIDS2018"])

# Model Selection Dropdown
model_type = st.sidebar.selectbox("Select Model", ["CNN", "LSTM", "Transformer"])

st.sidebar.info(
    "**Status**: Online\n\n"
    f"**Dataset**: {dataset}\n\n"
    f"**Model**: {model_type}\n\n"
    "**Device**: " + ("CUDA 🟢" if torch.cuda.is_available() else "CPU 🟠")
)

# Load Resources
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_data
def get_cached_feature_columns(_dataset):
    return utils.load_feature_columns(_dataset)


try:
    feature_cols = get_cached_feature_columns(dataset)
    # New signature: load_model_and_scaler(model_name, dataset_name, device)
    model, scaler, encoders = utils.load_model_and_scaler(model_type, dataset, device)
    model_loaded = model is not None
    if not model_loaded:
        st.sidebar.error(
            f"Failed to load {model_type} model for {dataset}. "
            f"Check if model file exists in results/models/."
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
        if dataset == "CICIDS2018":
            acc = "96.43%"
            dataset_label = "CICIDS2018"
        else:
            acc = "88.78%"
            dataset_label = "NSL-KDD"

        st.markdown(
            f"""
        <div class="card">
            <h4>Model Accuracy</h4>
            <div class="metric-value">{acc}</div>
            <p>On {dataset_label} Test Set</p>
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

    if dataset == "CICIDS2018":
        chart_data = pd.DataFrame(
            {
                "Model": ["CNN", "LSTM", "Transformer", "Autoencoder"],
                "Accuracy": [96.43, 95.90, 96.05, 95.0],
                "F1-Score": [0.96, 0.96, 0.96, 0.95],
            }
        )
    else:
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
        title=f"Comparative Analysis of Deep Learning Models ({dataset})",
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

    st.markdown(f"Enter network flow parameters to classify traffic. (**{dataset}** mode)")

    if dataset == "NSL-KDD":
        # NSL-KDD Live Prediction Form
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

            with st.expander("Advanced Network Features"):
                serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0)
                rerror_rate = st.slider("REJ Error Rate", 0.0, 1.0, 0.0)
                same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0)

            submitted = st.form_submit_button("Analyze Traffic")

        if submitted:
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
            X_scaled = utils.preprocess_input(df, scaler, feature_cols, None, dataset)
            X_tensor = torch.FloatTensor(X_scaled).to(device)

            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()

            st.markdown("### Analysis Result")

            if pred_class == 1:
                st.error(f"🚨 **Threat Detected: ATTACK**")
                st.metric("Confidence", f"{confidence*100:.2f}%")
            else:
                st.success(f"✅ **Traffic Status: NORMAL**")
                st.metric("Confidence", f"{confidence*100:.2f}%")

            st.bar_chart(df[["src_bytes", "dst_bytes", "count"]].T)

    elif dataset == "CICIDS2018":
        # CICIDS2018 Live Prediction Form
        with st.form("cicids_prediction_form"):
            st.markdown("#### CICIDS2018 Network Flow Features")

            col1, col2, col3 = st.columns(3)

            with col1:
                flow_duration = st.number_input("Flow Duration", min_value=0, value=0)
                tot_fwd_pkts = st.number_input("Total Fwd Packets", min_value=0, value=1)
                tot_bwd_pkts = st.number_input("Total Bwd Packets", min_value=0, value=0)
                fwd_pkt_len_mean = st.number_input("Fwd Pkt Len Mean", min_value=0.0, value=100.0)

            with col2:
                bwd_pkt_len_mean = st.number_input("Bwd Pkt Len Mean", min_value=0.0, value=0.0)
                flow_byts_s = st.number_input("Flow Bytes/s", min_value=0.0, value=1000.0)
                flow_pkts_s = st.number_input("Flow Packets/s", min_value=0.0, value=10.0)
                flow_iat_mean = st.number_input("Flow IAT Mean", min_value=0.0, value=0.0)

            with col3:
                fwd_iat_mean = st.number_input("Fwd IAT Mean", min_value=0.0, value=0.0)
                bwd_iat_mean = st.number_input("Bwd IAT Mean", min_value=0.0, value=0.0)
                pkt_len_mean = st.number_input("Pkt Len Mean", min_value=0.0, value=50.0)
                down_up_ratio = st.number_input("Down/Up Ratio", min_value=0, value=0)

            submitted = st.form_submit_button("Analyze Traffic")

        if submitted:
            # Build a row with all features, defaulting most to 0
            input_data = {}
            if feature_cols is not None:
                for col in feature_cols:
                    input_data[col] = 0.0

            # Set the values we collected
            field_mappings = {
                "Flow Duration": flow_duration,
                "Tot Fwd Pkts": tot_fwd_pkts,
                "Tot Bwd Pkts": tot_bwd_pkts,
                "Fwd Pkt Len Mean": fwd_pkt_len_mean,
                "Bwd Pkt Len Mean": bwd_pkt_len_mean,
                "Flow Byts/s": flow_byts_s,
                "Flow Pkts/s": flow_pkts_s,
                "Flow IAT Mean": flow_iat_mean,
                "Fwd IAT Mean": fwd_iat_mean,
                "Bwd IAT Mean": bwd_iat_mean,
                "Pkt Len Mean": pkt_len_mean,
                "Down/Up Ratio": down_up_ratio,
            }
            for key, value in field_mappings.items():
                if key in input_data:
                    input_data[key] = value

            df = pd.DataFrame([input_data])
            X_scaled = utils.preprocess_cicids2018_input(df, scaler, feature_cols)
            X_tensor = torch.FloatTensor(X_scaled).to(device)

            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()

            st.markdown("### Analysis Result")

            if pred_class == 1:
                st.error(f"🚨 **Threat Detected: ATTACK**")
                st.metric("Confidence", f"{confidence*100:.2f}%")
            else:
                st.success(f"✅ **Traffic Status: NORMAL**")
                st.metric("Confidence", f"{confidence*100:.2f}%")

# ==============================================================================
# PAGE: Batch Analysis
# ==============================================================================
elif page == "Batch Analysis":
    st.header("📂 Batch File Analysis")
    st.markdown(f"**Dataset mode:** {dataset}")

    uploaded_file = st.file_uploader(
        "Upload CSV / PCAP (Pre-processed)", type=["csv", "txt"]
    )

    if uploaded_file and model_loaded:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} samples.")

            if st.button("Run Predictions"):
                progress = st.progress(0)

                if dataset == "NSL-KDD":
                    # NSL-KDD preprocessing
                    missing_cols = [
                        c for c in utils.NSL_KDD_COLUMNS[:-2] if c not in df.columns
                    ]
                    if len(missing_cols) > 20:
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file,
                            header=None,
                            names=utils.NSL_KDD_COLUMNS,
                        )

                    X_scaled = utils.preprocess_input(
                        df, scaler, feature_cols, None, dataset
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
                        outputs = model(X_batch)
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        progress.progress(min(1.0, (i + batch_size) / len(X_scaled)))

                # Re-read original file for display (since preprocessing modifies df)
                uploaded_file.seek(0)
                df_display = pd.read_csv(uploaded_file)
                # Trim to match processed rows (NaN rows may have been dropped)
                df_display = df_display.head(len(all_preds))

                df_display["Prediction"] = [
                    "Attack" if p == 1 else "Normal" for p in all_preds
                ]
                df_display["Attack_Probability"] = all_probs

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
                st.dataframe(df_display.head(100))

                # Download
                csv = df_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV", csv, "results.csv", "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
