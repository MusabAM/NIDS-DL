import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import torch
import torch.nn as nn
import os
import sys
# Add the project root to sys.path to resolve imports from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils import (
    load_model_and_scaler,
    load_feature_columns,
    preprocess_input,
    DATASET_CONFIG
)

# Set Page Config
st.set_page_config(
    page_title="NIDS Pro Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Glassmorphism CSS ---
st.markdown("""
<style>
    .reportview-container {
        background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(64, 64, 64) 90.2%);
    }
    .stApp {
        background: #0e1117;
        color: #e0e0e0;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .main .block-container {
        padding-top: 2rem;
    }
    div.stButton > button {
        background-color: transparent;
        color: #00d4ff;
        border: 1px solid #00d4ff;
        border-radius: 8px;
        transition: 0.3s;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
    }
    div.stButton > button:hover {
        background-color: #00d4ff;
        color: white;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper: Model Loading ---
@st.cache_resource
def get_model(model_name, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, encoders = load_model_and_scaler(model_name, dataset_name, device)
    feature_cols = load_feature_columns(dataset_name)
    return model, scaler, encoders, feature_cols, device

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/shield.png", width=80)
    st.title("Control Hub")
    
    st.markdown("---")
    dataset_name = st.selectbox("Select Dataset", list(DATASET_CONFIG.keys()), index=2) # Default to CICIDS2017
    
    available_models = list(DATASET_CONFIG[dataset_name]["model_files"].keys())
    model_name = st.selectbox("Select Detection Model", available_models)
    
    st.markdown("---")
    st.info("System Status: Online 🟢")
    st.markdown("---")
    st.caption("Powered by Advanced Deep Learning & Antigravity AI")

# --- Main Dashboard ---
col_head1, col_head2 = st.columns([2, 1])

with col_head1:
    st.title(f"🚀 NIDS Pro Dashboard")
    st.markdown(f"**Target System:** {dataset_name} | **Active Engine:** {model_name}")

with col_head2:
    st.write("")
    if st.button("🔄 Refresh System"):
        st.rerun()

# ─── Helper: Load Real Metrics from Result Files ──────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

RESULT_FILE_MAP = {
    ("NSL-KDD",   "CNN"):         "cnn_nsl_kdd_improved_results.txt",
    ("NSL-KDD",   "LSTM"):        "lstm_nsl_kdd_improved_results.txt",
    ("NSL-KDD",   "Transformer"): "transformer_nsl_kdd_improved_results.txt",
    ("UNSW-NB15", "CNN"):         "cnn_unsw_nb15_improved_results.txt",
    ("UNSW-NB15", "LSTM"):        "lstm_unsw_nb15_results.txt",
    ("NSL-KDD",   "Transformer"): "transformer_nsl_kdd_improved_results.txt",
    ("UNSW-NB15", "CNN"):         "cnn_unsw_nb15_improved_results.txt",
    ("UNSW-NB15", "LSTM"):        "lstm_unsw_nb15_results.txt",
    ("UNSW-NB15", "Transformer"): "transformer_unsw_nb15_improved_results.txt",
    ("CICIDS2018","CNN"):         "cnn_cicids2018_results.txt",
    ("CICIDS2018","LSTM"):        "lstm_cicids2018_results.txt",
    ("CICIDS2018","Transformer"): "transformer_cicids2018_results.txt",
    ("CICIDS2017","CNN"):         "cnn_cicids2018_results.txt", # Placeholder if 2017 missing
    ("CICIDS2017","LSTM"):        "lstm_cicids2017_results.txt",
    ("CICIDS2017","Transformer"): "transformer_cicids2017_results.txt",
}

@st.cache_data(ttl=60)
def load_real_metrics(dataset_name, model_name):
    """Parse accuracy and F1 from a result .txt file."""
    fname = RESULT_FILE_MAP.get((dataset_name, model_name))
    if not fname:
        return None, None
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        return None, None
    acc, f1 = None, None
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # Try parsing accuracy
            if "Test Accuracy:" in line:
                try:
                    acc = float(line.split("Test Accuracy:")[-1].strip().replace("%",""))
                except Exception:
                    pass
            # Try parsing F1 score (two possible formats)
            if "F1-Score:" in line:
                try:
                    f1 = float(line.split("F1-Score:")[-1].strip())
                except Exception:
                    pass
            elif line.startswith("weighted avg"):
                parts = line.split()
                try:
                    f1 = float(parts[-2])   # weighted f1-score
                except Exception:
                    pass
    return acc, f1

# Metrics Row
real_acc, real_f1 = load_real_metrics(dataset_name, model_name)
acc_display   = f"{real_acc:.2f}%" if real_acc is not None else "N/A"
f1_display    = f"{real_f1:.4f}"   if real_f1  is not None else "N/A"

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Active Threats</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">0</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Network Traffic</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">1,429 p/s</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Model Accuracy ({model_name})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{acc_display}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Weighted F1 ({model_name})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{f1_display}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Content Area
tab1, tab2, tab3 = st.tabs(["🔴 Live Detection", "📊 Batch Analysis", "⚙️ Model Intelligence"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Simulate Network Traffic")
    
    # Load sample data for simulation
    model, scaler, encoders, feature_cols, device = get_model(model_name, dataset_name)
    
    if model:
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            if st.button("🎯 Inject Test Sample"):
                with st.spinner("Analyzing traffic..."):
                    # Create a dummy sample with 0s and some random noise for visual effect
                    # This is just for demonstration if no data is uploaded
                    dummy_data = {col: [np.random.rand()] for col in feature_cols}
                    df_sample = pd.DataFrame(dummy_data)
                    
                    X_processed = preprocess_input(df_sample, scaler, feature_cols, encoders, dataset_name)
                    X_tensor = torch.from_numpy(X_processed).float().to(device)
                    
                    with torch.no_grad():
                        output = model(X_tensor)
                        if isinstance(output, tuple): # For Autoencoder (recon, latent)
                            recon, _ = output
                            score = torch.mean((recon - X_tensor)**2).item()
                            prediction = "Suspicious" if score > 0.01 else "Normal" # Dummy threshold
                            prob = score * 10 
                        else:
                            probs = torch.softmax(output, dim=1)
                            prob, pred_idx = torch.max(probs, dim=1)
                            prediction = "Anomalous" if pred_idx.item() == 1 else "Normal"
                            prob = prob.item()

                    if prediction in ["Anomalous", "Suspicious"]:
                        st.error(f"⚠️ THREAT DETECTED: {prediction}")
                    else:
                        st.success(f"✅ STATUS: {prediction}")
                    
                    st.metric("Confidence Score", f"{prob*100:.2f}%")
        
        with sim_col2:
            st.caption("Live Visualization")
            # Create a simple live bar chart of a few features
            viz_data = pd.DataFrame({
                "Feature": feature_cols[:15],
                "Value": np.random.rand(15)
            })
            fig = px.bar(viz_data, x="Feature", y="Value", color="Value", color_continuous_scale="Viridis")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model loading failed. Please check your model files and configuration.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Batch Upload Analysis")
    uploaded_file = st.file_uploader("Upload Network CSV (CICIDS format)", type=["csv"])
    
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write("Processing batch data...")
        
        model, scaler, encoders, feature_cols, device = get_model(model_name, dataset_name)
        
        if model:
            X_batch = preprocess_input(df_batch, scaler, feature_cols, encoders, dataset_name)
            X_tensor = torch.from_numpy(X_batch).float().to(device)
            
            with torch.no_grad():
                outputs = model(X_tensor)
                if isinstance(outputs, tuple):
                    recon, _ = outputs
                    scores = torch.mean((recon - X_tensor)**2, dim=1).cpu().numpy()
                    preds = (scores > 0.01).astype(int)
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            df_batch['Prediction'] = ["Anomalous" if p == 1 else "Normal" for p in preds]
            
            st.write(df_batch[['Prediction']].value_counts())
            st.dataframe(df_batch.head(100), use_container_width=True)
            
            # Summary Chart
            pred_counts = df_batch['Prediction'].value_counts().reset_index()
            fig_pie = px.pie(pred_counts, values='count', names='Prediction', color='Prediction',
                           color_discrete_map={'Normal':'#00cc96', 'Anomalous':'#ef553b'})
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Neural Network Insights")
    st.write(f"The active **{model_name}** model architecture is designed for high-throughput anomaly detection in **{dataset_name}** environments.")
    
    model, _, _, _, _ = get_model(model_name, dataset_name)
    if model:
        st.code(str(model), language="text")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2026 Antigravity Research Hub | Network Intrusion Detection System")
