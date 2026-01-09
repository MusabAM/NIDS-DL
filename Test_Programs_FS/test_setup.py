# Test importing project modules
import sys
sys.path.insert(0, '.')

print("Testing imports...")

# Test PyTorch model
from src.models.classical.cnn import CNNClassifier
print("✓ CNNClassifier imported")

# Test LSTM
from src.models.classical.lstm import LSTMClassifier
print("✓ LSTMClassifier imported")

# Test data loading
from src.data.datasets import NIDSDataset, get_dataloaders
print("✓ Data utilities imported")

# Create a simple model
import torch
model = CNNClassifier(input_dim=41, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"✓ Model created and moved to {device}")

# Test forward pass
x = torch.randn(32, 41).to(device)
output = model(x)
print(f"✓ Forward pass successful: input shape {x.shape}, output shape {output.shape}")

print("\n🎉 All tests passed! Environment is ready.")
