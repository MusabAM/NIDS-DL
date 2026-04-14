import pennylane as qml
print("Available devices:")
for device in qml.plugin_devices:
    print(f"- {device}")

try:
    dev = qml.device("lightning.qubit", wires=1)
    print("lightning.qubit is SUPPORTED")
except Exception as e:
    print(f"lightning.qubit is NOT supported/installed: {e}")

try:
    dev = qml.device("lightning.gpu", wires=1)
    print("lightning.gpu is SUPPORTED")
except Exception as e:
    print(f"lightning.gpu is NOT supported/installed: {e}")
