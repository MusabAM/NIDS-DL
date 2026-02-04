"""
Hilbert Curve Visualization for GPU Performance Testing
Space-filling curve that maps 1D data to 2D space while preserving locality.

Useful for:
- Visualizing high-dimensional network traffic patterns
- Testing GPU computational capabilities
- Creating 2D representations of model outputs
"""

import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration available")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU (install cupy-cuda12x for GPU support)")


class HilbertCurve:
    """
    Generate and visualize Hilbert space-filling curves.
    Supports both CPU and GPU computation.
    """

    def __init__(self, order: int = 5, use_gpu: bool = True):
        """
        Initialize Hilbert Curve generator.

        Args:
            order: Order of the Hilbert curve (n). The curve will have 2^n x 2^n points.
            use_gpu: Whether to use GPU acceleration (requires CuPy)
        """
        self.order = order
        self.n_points = 2**order
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        print(
            f"Initialized Hilbert Curve (order={order}, size={self.n_points}x{self.n_points})"
        )
        print(f"Computing on: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")

    def d2xy(self, d: int) -> Tuple[int, int]:
        """
        Convert distance along Hilbert curve to (x, y) coordinates.

        Args:
            d: Distance along the curve (0 to n_points^2 - 1)

        Returns:
            Tuple of (x, y) coordinates
        """
        n = self.n_points
        x = y = 0
        s = 1

        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            x, y = self._rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2

        return x, y

    def xy2d(self, x: int, y: int) -> int:
        """
        Convert (x, y) coordinates to distance along Hilbert curve.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Distance along the curve
        """
        n = self.n_points
        d = 0
        s = n // 2

        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = self._rot(s, x, y, rx, ry)
            s //= 2

        return d

    @staticmethod
    def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        """Rotate/flip a quadrant appropriately."""
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    def generate_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the complete Hilbert curve coordinates.

        Returns:
            Tuple of (x_coords, y_coords) arrays
        """
        start_time = time.time()

        total_points = self.n_points**2
        x_coords = []
        y_coords = []

        for d in range(total_points):
            x, y = self.d2xy(d)
            x_coords.append(x)
            y_coords.append(y)

        # Convert to numpy arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        if self.use_gpu:
            # Transfer to GPU if needed
            x_coords = self.xp.array(x_coords)
            y_coords = self.xp.array(y_coords)

        elapsed = time.time() - start_time
        print(
            f"Generated {total_points:,} points in {elapsed:.4f}s "
            f"({'GPU' if self.use_gpu else 'CPU'})"
        )

        return x_coords, y_coords

    def map_data_to_curve(self, data: np.ndarray) -> np.ndarray:
        """
        Map 1D data array to 2D Hilbert curve space.

        Args:
            data: 1D array of values to map

        Returns:
            2D array representing the data mapped to Hilbert curve
        """
        start_time = time.time()

        # Ensure data fits in the curve
        total_points = self.n_points**2
        if len(data) > total_points:
            print(
                f"Warning: Data length ({len(data)}) exceeds curve capacity "
                f"({total_points}). Truncating..."
            )
            data = data[:total_points]
        elif len(data) < total_points:
            # Pad with zeros
            padding = total_points - len(data)
            data = np.concatenate([data, np.zeros(padding)])

        # Transfer to GPU if needed
        if self.use_gpu:
            data = self.xp.array(data)

        # Create 2D grid
        grid = self.xp.zeros((self.n_points, self.n_points))

        # Map data to curve
        for d, value in enumerate(data):
            x, y = self.d2xy(d)
            grid[y, x] = value

        # Transfer back to CPU if needed
        if self.use_gpu:
            grid = cp.asnumpy(grid)

        elapsed = time.time() - start_time
        print(f"Mapped {len(data):,} values to 2D grid in {elapsed:.4f}s")

        return grid

    def visualize_curve(
        self,
        save_path: Optional[str] = None,
        show_points: bool = True,
        figsize: Tuple[int, int] = (12, 12),
    ):
        """
        Visualize the Hilbert curve.

        Args:
            save_path: Optional path to save the figure
            show_points: Whether to show individual points
            figsize: Figure size
        """
        x_coords, y_coords = self.generate_curve()

        # Transfer to CPU for plotting
        if self.use_gpu:
            x_coords = cp.asnumpy(x_coords)
            y_coords = cp.asnumpy(y_coords)

        plt.figure(figsize=figsize)
        plt.plot(x_coords, y_coords, "b-", linewidth=0.5, alpha=0.7)

        if show_points and self.order <= 4:
            plt.scatter(
                x_coords,
                y_coords,
                c=range(len(x_coords)),
                cmap="viridis",
                s=20,
                alpha=0.6,
            )
            plt.colorbar(label="Position along curve")

        plt.title(f"Hilbert Curve (Order {self.order})", fontsize=16, fontweight="bold")
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        plt.show()

    def visualize_data(
        self,
        data: np.ndarray,
        title: str = "Data on Hilbert Curve",
        cmap: str = "viridis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 12),
    ):
        """
        Visualize data mapped to Hilbert curve.

        Args:
            data: 1D array of values to visualize
            title: Plot title
            cmap: Colormap to use
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        grid = self.map_data_to_curve(data)

        plt.figure(figsize=figsize)
        im = plt.imshow(grid, cmap=cmap, interpolation="nearest", origin="lower")
        plt.colorbar(im, label="Value")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        plt.show()


def benchmark_gpu_performance(orders: List[int] = [5, 6, 7, 8]) -> dict:
    """
    Benchmark GPU vs CPU performance for Hilbert curve generation.

    Args:
        orders: List of curve orders to test

    Returns:
        Dictionary with benchmark results
    """
    results = {"orders": orders, "cpu_times": [], "gpu_times": [], "speedups": []}

    print("=" * 60)
    print("GPU Performance Benchmark")
    print("=" * 60)

    for order in orders:
        print(
            f"\nTesting order {order} ({2**order}x{2**order} = {(2**order)**2:,} points)"
        )

        # CPU benchmark
        print("  CPU: ", end="", flush=True)
        hc_cpu = HilbertCurve(order=order, use_gpu=False)
        start = time.time()
        _ = hc_cpu.generate_curve()
        cpu_time = time.time() - start
        results["cpu_times"].append(cpu_time)
        print(f"{cpu_time:.4f}s")

        if GPU_AVAILABLE:
            # GPU benchmark
            print("  GPU: ", end="", flush=True)
            hc_gpu = HilbertCurve(order=order, use_gpu=True)
            start = time.time()
            _ = hc_gpu.generate_curve()
            gpu_time = time.time() - start
            results["gpu_times"].append(gpu_time)
            speedup = cpu_time / gpu_time
            results["speedups"].append(speedup)
            print(f"{gpu_time:.4f}s (Speedup: {speedup:.2f}x)")
        else:
            results["gpu_times"].append(None)
            results["speedups"].append(None)
            print("  GPU: Not available")

    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    for i, order in enumerate(orders):
        print(f"Order {order}: CPU={results['cpu_times'][i]:.4f}s", end="")
        if results["gpu_times"][i]:
            print(
                f", GPU={results['gpu_times'][i]:.4f}s, "
                f"Speedup={results['speedups'][i]:.2f}x"
            )
        else:
            print(", GPU=N/A")

    return results


def visualize_model_predictions(
    predictions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    order: int = 6,
    use_gpu: bool = True,
    save_dir: Optional[str] = None,
):
    """
    Visualize model predictions using Hilbert curve.

    Args:
        predictions: 1D array of model predictions (0=Normal, 1=Attack)
        labels: Optional 1D array of true labels
        order: Hilbert curve order
        use_gpu: Whether to use GPU
        save_dir: Optional directory to save visualizations
    """
    hc = HilbertCurve(order=order, use_gpu=use_gpu)

    print("\n" + "=" * 60)
    print("Model Prediction Visualization")
    print("=" * 60)

    # Visualize predictions
    save_path = f"{save_dir}/predictions_hilbert.png" if save_dir else None
    hc.visualize_data(
        predictions,
        title="Model Predictions on Hilbert Curve (0=Normal, 1=Attack)",
        cmap="RdYlGn_r",
        save_path=save_path,
        figsize=(14, 12),
    )

    # Visualize labels if provided
    if labels is not None:
        save_path = f"{save_dir}/labels_hilbert.png" if save_dir else None
        hc.visualize_data(
            labels,
            title="True Labels on Hilbert Curve (0=Normal, 1=Attack)",
            cmap="RdYlGn_r",
            save_path=save_path,
            figsize=(14, 12),
        )

        # Visualize errors
        errors = (predictions != labels).astype(float)
        save_path = f"{save_dir}/errors_hilbert.png" if save_dir else None
        hc.visualize_data(
            errors,
            title="Prediction Errors on Hilbert Curve (Red=Error, Green=Correct)",
            cmap="RdYlGn_r",
            save_path=save_path,
            figsize=(14, 12),
        )


# Example usage
if __name__ == "__main__":
    print("Hilbert Curve GPU Visualization Tool")
    print("=" * 60)

    # Example 1: Generate and visualize basic Hilbert curve
    print("\n1. Generating basic Hilbert curve...")
    hc = HilbertCurve(order=5, use_gpu=True)
    hc.visualize_curve(save_path="results/hilbert_curve_order5.png")

    # Example 2: Visualize random data
    print("\n2. Visualizing random data on Hilbert curve...")
    random_data = np.random.rand(1024)  # Random values
    hc.visualize_data(
        random_data,
        title="Random Data on Hilbert Curve",
        save_path="results/hilbert_random_data.png",
    )

    # Example 3: Benchmark GPU performance
    print("\n3. Running GPU performance benchmark...")
    benchmark_results = benchmark_gpu_performance(orders=[4, 5, 6, 7])

    # Example 4: Visualize synthetic predictions
    print("\n4. Visualizing synthetic model predictions...")
    n_samples = 2048
    synthetic_predictions = np.random.randint(0, 2, n_samples)
    synthetic_labels = np.random.randint(0, 2, n_samples)

    visualize_model_predictions(
        synthetic_predictions,
        synthetic_labels,
        order=6,
        use_gpu=True,
        save_dir="results",
    )

    print("\n" + "=" * 60)
    print("Examples completed! Check the 'results' directory for outputs.")
    print("=" * 60)
