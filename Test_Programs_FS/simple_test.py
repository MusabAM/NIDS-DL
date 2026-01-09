"""Simple test to print LSTM model metrics"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from src.data import get_dataset, get_dataloaders
from src.models import LSTMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load data
print("Loading dataset...", flush=True)
data = get_dataset(name='nsl_kdd', classification='binary', normalize='standard', handle_imbalance='none')
_, _, test_loader = get_dataloaders(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'], batch_size=256)

# Load model
print("Loading model...", flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_dim=data['info'].num_features, num_classes=data['info'].num_classes, lstm_units=[128, 64], dense_units=[128, 64], bidirectional=True, dropout_rate=0.3)
checkpoint = torch.load('./results/models/lstm_nsl_kdd_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Get predictions
print("Running inference...", flush=True)
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X.to(device))
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())

y_true, y_pred = np.array(all_labels), np.array(all_preds)

# Write results
with open('results/test_results.txt', 'w') as f:
    f.write('='*60 + '\n')
    f.write('  LSTM Model Test Results - NSL-KDD Dataset\n')
    f.write('='*60 + '\n')
    f.write(f'Device: {device}\n')
    f.write(f'Test Samples: {len(y_true):,}\n')
    f.write('='*60 + '\n')
    f.write(f'Accuracy:  {accuracy_score(y_true, y_pred)*100:.2f}%\n')
    f.write(f'Precision: {precision_score(y_true, y_pred)*100:.2f}%\n')
    f.write(f'Recall:    {recall_score(y_true, y_pred)*100:.2f}%\n')
    f.write(f'F1 Score:  {f1_score(y_true, y_pred)*100:.2f}%\n')
    f.write('='*60 + '\n\n')
    f.write('Classification Report:\n')
    f.write(classification_report(y_true, y_pred, target_names=data['info'].class_names))

print("Results saved to results/test_results.txt", flush=True)
