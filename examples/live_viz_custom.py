"""
Example: Custom data generator for live Hilbert map
Shows how to integrate your own data source
"""

import time

import numpy as np

from src.visualization.live_hilbert_map import LiveHilbertMap

# Create visualization
viz = LiveHilbertMap(
    order=7,  # 128x128 grid
    cell_size=4,  # 4px per cell
    fps=60,  # 60 FPS
    title="Custom NIDS Live Monitor",
)


def custom_data_generator():
    """
    Custom generator function.

    Yields:
        (prediction, label) tuples
        - prediction: 0 = normal, 1 = attack
        - label: optional true label (can be None)
    """

    # Example 1: Simulate network traffic with attack bursts
    total_samples = viz.grid_size**2

    for i in range(total_samples):
        # Normal traffic baseline (85%)
        if np.random.random() < 0.85:
            prediction = 0
        else:
            prediction = 1

        # Simulate attack bursts
        if 2000 <= i < 2500:  # Burst 1: DDoS
            prediction = 1 if np.random.random() < 0.9 else 0
        elif 6000 <= i < 6200:  # Burst 2: Port scan
            prediction = 1 if np.random.random() < 0.95 else 0
        elif 10000 <= i < 10500:  # Burst 3: Intrusion attempt
            prediction = 1 if np.random.random() < 0.8 else 0

        # Yield prediction
        yield prediction, None

        # Optional: Add delay to control speed
        # time.sleep(0.001)  # 1ms delay = ~1000 samples/sec


print("🚀 Starting custom data visualization...")
print("=" * 60)
print("This demo simulates network traffic with attack bursts")
print("Watch for RED clusters appearing at different times")
print("=" * 60)
print("\nControls:")
print("  R - Reset visualization")
print("  ESC - Quit")
print()

# Run visualization with custom generator
viz.run(custom_data_generator)

print("\n✅ Visualization closed")
