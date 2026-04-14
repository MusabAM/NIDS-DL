import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.quantum.pennylane_models import HybridQuantumClassifier

def test_device(device_name):
    print(f"Testing device: {device_name}")
    try:
        model = HybridQuantumClassifier(
            input_dim=8,
            num_classes=2,
            n_qubits=8,
            n_quantum_layers=2,
            device=device_name
        )
        print(f"Successfully initialized model with {device_name}")
        
        # Test forward pass
        x = torch.randn(1, 8)
        output = model(x)
        print(f"Forward pass output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"Failed to initialize model with {device_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test default.qubit
    success_default = test_device("default.qubit")
    
    # Test lightning.qubit
    success_lightning = test_device("lightning.qubit")
    
    if success_default and success_lightning:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
