import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

file_path = 'data/raw/nsl-kdd/train.txt'
if not os.path.exists(file_path):
    print(f"Training file not found at {file_path}")
    exit(1)

df = pd.read_csv(file_path, header=None, names=columns)
categorical_cols = ['protocol_type', 'service', 'flag']
df_encoded = pd.get_dummies(df.drop(['label', 'difficulty_level'], axis=1), columns=categorical_cols)

# Align columns with what the dashboard expects (sorted feature names)
feature_cols = sorted(df_encoded.columns.tolist())
X = df_encoded[feature_cols].values

scaler = StandardScaler()
scaler.fit(X)

save_dir = 'results/models'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'cnn_scaler.pkl')

with open(save_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Scaler regenerated successfully at {save_path}")
print(f"Feature count: {len(feature_cols)}")
