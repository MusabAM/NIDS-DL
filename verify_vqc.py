import os
import sys
import torch
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from backend.utils import load_model_and_scaler, preprocess_input, DATASET_CONFIGS

dataset = "NSL-KDD"
model_type = "VQC"
device = torch.device("cpu") # Use CPU for verification

print(f"Verifying {model_type} for {dataset}...")

try:
    # 1. Load Model and Preprocessing
    model, scaler, encoders = load_model_and_scaler(model_type, dataset, device)
    
    if model is None:
        print("FAILED: Model or scaler could not be loaded.")
        sys.exit(1)
    
    print("SUCCESS: Model and scaler loaded successfully.")
    
    # Check if PCA was attached
    if hasattr(scaler, "pca_transformer"):
        print("SUCCESS: PCA transformer detected in scaler.")
    else:
        print("FAILED: PCA transformer NOT found in scaler.")
        sys.exit(1)

    # 2. Create Dummy Input
    # NSL-KDD features template
    features = {
        'duration': 0, 'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
        'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
        'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
        'is_guest_login': 0, 'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
        'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 255, 'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF'
    }
    df = pd.DataFrame([features])

    # 3. Preprocess Input
    from backend.utils import load_feature_columns
    feature_cols = load_feature_columns(dataset)
    X_scaled = preprocess_input(df, scaler, feature_cols, None, dataset, model_type=model_type)
    
    print(f"X_scaled shape: {X_scaled.shape}")
    if X_scaled.shape[1] != 8:
        print(f"FAILED: Expected 8 features after PCA, got {X_scaled.shape[1]}")
        sys.exit(1)
    
    # 4. Forward Pass
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
    print(f"Inferece SUCCESS. Prediction: {pred}, Confidence: {probs[0][pred].item():.4f}")

except Exception as e:
    print(f"ERROR during verification: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
