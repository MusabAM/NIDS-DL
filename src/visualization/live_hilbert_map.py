"""
Live Hilbert Map Visualization with Pygame
Real-time visualization of network intrusion detection using space-filling curves.

Author: NIDS-DL Project
"""

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pygame

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# Try to import TensorFlow for model inference
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Model inference disabled.")


class HilbertMapper:
    """Fast Hilbert curve coordinate mapper."""

    def __init__(self, order: int):
        self.order = order
        self.n = 2**order
        self._build_lookup_table()

    def _build_lookup_table(self):
        """Pre-compute Hilbert coordinates for faster lookup."""
        total_points = self.n**2
        self.lookup_x = np.zeros(total_points, dtype=np.int32)
        self.lookup_y = np.zeros(total_points, dtype=np.int32)

        for d in range(total_points):
            x, y = self.d2xy(d)
            self.lookup_x[d] = x
            self.lookup_y[d] = y

    def d2xy(self, d: int) -> Tuple[int, int]:
        """Convert distance to (x, y) coordinates."""
        n = self.n
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

    def get_coords(self, index: int) -> Tuple[int, int]:
        """Fast coordinate lookup."""
        if index < len(self.lookup_x):
            return self.lookup_x[index], self.lookup_y[index]
        return self.d2xy(index)

    @staticmethod
    def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        """Rotate/flip quadrant."""
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y


class LiveHilbertMap:
    """
    Real-time Hilbert map visualization using Pygame.
    Displays model predictions as they are processed.
    """

    def __init__(
        self,
        order: int = 7,
        cell_size: int = 4,
        fps: int = 60,
        title: str = "Live NIDS Hilbert Map",
    ):
        """
        Initialize live visualization.

        Args:
            order: Hilbert curve order (grid will be 2^order x 2^order)
            cell_size: Size of each cell in pixels
            fps: Target frames per second
            title: Window title
        """
        pygame.init()

        self.order = order
        self.grid_size = 2**order
        self.cell_size = cell_size
        self.fps = fps
        self.title = title

        # Calculate window dimensions
        self.map_width = self.grid_size * cell_size
        self.map_height = self.grid_size * cell_size
        self.info_panel_width = 300
        self.window_width = self.map_width + self.info_panel_width
        self.window_height = self.map_height

        # Initialize pygame window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(title)

        # Initialize Hilbert mapper
        self.hilbert = HilbertMapper(order)

        # Initialize data grid (0=unknown, 1=normal, 2=attack)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Color scheme
        self.colors = {
            "unknown": (20, 20, 30),  # Dark gray
            "normal": (50, 200, 50),  # Green
            "attack": (255, 50, 50),  # Red
            "background": (15, 15, 20),  # Dark background
            "text": (200, 200, 200),  # Light gray text
            "panel": (25, 25, 35),  # Panel background
            "border": (60, 60, 80),  # Border color
        }

        # Statistics
        self.total_processed = 0
        self.normal_count = 0
        self.attack_count = 0
        self.current_index = 0
        self.start_time = time.time()

        # Performance metrics
        self.clock = pygame.time.Clock()
        self.actual_fps = 0

        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # Data queue for threaded processing
        self.data_queue = queue.Queue(maxsize=1000)
        self.running = True

        print(f"Initialized Live Hilbert Map:")
        print(f"  Grid: {self.grid_size}x{self.grid_size} ({self.grid_size**2} points)")
        print(f"  Window: {self.window_width}x{self.window_height}")
        print(f"  Cell size: {cell_size}px")
        print(f"  Target FPS: {fps}")

    def add_prediction(self, prediction: int, label: Optional[int] = None):
        """
        Add a new prediction to the map.

        Args:
            prediction: 0 for normal, 1 for attack
            label: Optional true label for accuracy tracking
        """
        if self.current_index >= self.grid_size**2:
            # Map is full, restart
            self.reset()

        x, y = self.hilbert.get_coords(self.current_index)

        # Map prediction to color code (1=normal, 2=attack)
        self.grid[y, x] = 1 if prediction == 0 else 2

        # Update statistics
        self.total_processed += 1
        if prediction == 0:
            self.normal_count += 1
        else:
            self.attack_count += 1

        self.current_index += 1

    def add_prediction_async(self, prediction: int, label: Optional[int] = None):
        """Thread-safe prediction addition."""
        try:
            self.data_queue.put((prediction, label), block=False)
        except queue.Full:
            pass  # Skip if queue is full

    def process_queue(self):
        """Process queued predictions."""
        processed = 0
        max_per_frame = 100  # Process up to 100 predictions per frame

        while processed < max_per_frame and not self.data_queue.empty():
            try:
                prediction, label = self.data_queue.get_nowait()
                self.add_prediction(prediction, label)
                processed += 1
            except queue.Empty:
                break

    def reset(self):
        """Reset the map."""
        self.grid.fill(0)
        self.current_index = 0
        self.total_processed = 0
        self.normal_count = 0
        self.attack_count = 0
        self.start_time = time.time()

    def draw_map(self):
        """Draw the Hilbert map."""
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                value = self.grid[y, x]

                if value == 0:
                    color = self.colors["unknown"]
                elif value == 1:
                    color = self.colors["normal"]
                else:
                    color = self.colors["attack"]

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)

    def draw_info_panel(self):
        """Draw the information panel."""
        panel_x = self.map_width

        # Panel background
        panel_rect = pygame.Rect(panel_x, 0, self.info_panel_width, self.window_height)
        pygame.draw.rect(self.screen, self.colors["panel"], panel_rect)
        pygame.draw.line(
            self.screen,
            self.colors["border"],
            (panel_x, 0),
            (panel_x, self.window_height),
            2,
        )

        y_offset = 20

        # Title
        title = self.font_medium.render("NIDS Monitor", True, self.colors["text"])
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 50

        # Statistics
        stats = [
            ("Total Processed", f"{self.total_processed:,}"),
            ("", ""),
            ("Normal Traffic", f"{self.normal_count:,}"),
            ("Attacks", f"{self.attack_count:,}"),
        ]

        for label, value in stats:
            if label:
                text = self.font_small.render(label, True, self.colors["text"])
                self.screen.blit(text, (panel_x + 20, y_offset))
                y_offset += 25

                value_text = self.font_medium.render(value, True, self.colors["text"])
                self.screen.blit(value_text, (panel_x + 30, y_offset))
                y_offset += 35
            else:
                y_offset += 10

        # Progress
        if self.grid_size**2 > 0:
            progress = (self.current_index / (self.grid_size**2)) * 100
            progress_text = self.font_small.render(
                f"Progress: {progress:.1f}%", True, self.colors["text"]
            )
            self.screen.blit(progress_text, (panel_x + 20, y_offset))
            y_offset += 30

            # Progress bar
            bar_width = 260
            bar_height = 20
            bar_x = panel_x + 20
            bar_y = y_offset

            # Background
            pygame.draw.rect(
                self.screen,
                self.colors["unknown"],
                (bar_x, bar_y, bar_width, bar_height),
            )

            # Progress
            progress_width = int(bar_width * progress / 100)
            pygame.draw.rect(
                self.screen,
                self.colors["normal"],
                (bar_x, bar_y, progress_width, bar_height),
            )

            # Border
            pygame.draw.rect(
                self.screen,
                self.colors["border"],
                (bar_x, bar_y, bar_width, bar_height),
                2,
            )

            y_offset += 40

        # Performance
        y_offset += 20
        perf_text = self.font_small.render("Performance", True, self.colors["text"])
        self.screen.blit(perf_text, (panel_x + 20, y_offset))
        y_offset += 30

        fps_text = self.font_small.render(
            f"FPS: {self.actual_fps:.1f}", True, self.colors["text"]
        )
        self.screen.blit(fps_text, (panel_x + 30, y_offset))
        y_offset += 25

        # Processing rate
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            rate = self.total_processed / elapsed
            rate_text = self.font_small.render(
                f"Rate: {rate:.1f} samples/s", True, self.colors["text"]
            )
            self.screen.blit(rate_text, (panel_x + 30, y_offset))
            y_offset += 35

        # Legend
        y_offset += 20
        legend_text = self.font_small.render("Legend", True, self.colors["text"])
        self.screen.blit(legend_text, (panel_x + 20, y_offset))
        y_offset += 30

        legend_items = [
            ("Unknown", self.colors["unknown"]),
            ("Normal", self.colors["normal"]),
            ("Attack", self.colors["attack"]),
        ]

        for label, color in legend_items:
            # Color box
            pygame.draw.rect(self.screen, color, (panel_x + 30, y_offset, 20, 20))
            pygame.draw.rect(
                self.screen, self.colors["border"], (panel_x + 30, y_offset, 20, 20), 1
            )

            # Label
            text = self.font_small.render(label, True, self.colors["text"])
            self.screen.blit(text, (panel_x + 60, y_offset + 2))
            y_offset += 30

        # Controls
        y_offset = self.window_height - 80
        controls_text = self.font_small.render("Controls", True, self.colors["text"])
        self.screen.blit(controls_text, (panel_x + 20, y_offset))
        y_offset += 25

        control_items = ["R - Reset", "ESC - Quit"]

        for item in control_items:
            text = self.font_small.render(item, True, self.colors["text"])
            self.screen.blit(text, (panel_x + 30, y_offset))
            y_offset += 20

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()

    def update(self):
        """Update the display."""
        self.handle_events()
        self.process_queue()

        # Clear screen
        self.screen.fill(self.colors["background"])

        # Draw components
        self.draw_map()
        self.draw_info_panel()

        # Update display
        pygame.display.flip()

        # Control framerate
        self.clock.tick(self.fps)
        self.actual_fps = self.clock.get_fps()

    def run(self, data_generator: Optional[Callable] = None):
        """
        Run the visualization loop.

        Args:
            data_generator: Optional function that yields (prediction, label) tuples
        """
        print("\nStarting Live Hilbert Map...")
        print("Controls: R=Reset, ESC=Quit\n")

        # Start data generator thread if provided
        if data_generator:
            thread = threading.Thread(
                target=self._run_data_generator, args=(data_generator,), daemon=True
            )
            thread.start()

        # Main loop
        while self.running:
            self.update()

        pygame.quit()
        print("\nVisualization closed.")

    def _run_data_generator(self, generator: Callable):
        """Run data generator in separate thread."""
        try:
            for prediction, label in generator():
                if not self.running:
                    break
                self.add_prediction_async(prediction, label)
                time.sleep(0.001)  # Small delay to prevent overwhelming the queue
        except Exception as e:
            print(f"Data generator error: {e}")


class ModelStreamer:
    """Stream predictions from a trained model."""

    def __init__(self, model_path: str, data_source: str):
        """
        Initialize model streamer.

        Args:
            model_path: Path to trained model (.h5 file)
            data_source: Path to test data CSV
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for model inference")

        self.model = tf.keras.models.load_model(model_path)
        self.data_source = data_source

        print(f"Loaded model: {model_path}")
        print(f"Data source: {data_source}")

    def stream_predictions(self, batch_size: int = 32, delay: float = 0.01):
        """
        Generator that yields predictions.

        Args:
            batch_size: Batch size for inference
            delay: Delay between batches (seconds)

        Yields:
            Tuple of (prediction, label)
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        # Load data
        print("Loading test data...")
        X_test = pd.read_csv(self.data_source.replace("y_test", "X_test"))

        try:
            y_test = pd.read_csv(self.data_source)
            if "label" in y_test.columns:
                y_test = y_test["label"].values
            else:
                y_test = y_test.values.flatten()
        except:
            y_test = None

        # Preprocess
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        print(f"Processing {len(X_test)} samples...")

        # Stream predictions
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test_scaled[i : i + batch_size]
            batch_pred = self.model.predict(batch_X, verbose=0)
            batch_pred_binary = (batch_pred > 0.5).astype(int).flatten()

            for j, pred in enumerate(batch_pred_binary):
                label = y_test[i + j] if y_test is not None else None
                yield int(pred), label

            time.sleep(delay)


# Example usage and demos
def demo_random_stream():
    """Demo with random data stream."""
    viz = LiveHilbertMap(order=7, cell_size=4, fps=60)

    def random_generator():
        """Generate random predictions."""
        while True:
            # Simulate normal traffic with occasional attacks
            prediction = np.random.choice([0, 1], p=[0.85, 0.15])
            yield prediction, prediction

    viz.run(random_generator)


def demo_pattern_stream():
    """Demo with pattern data."""
    viz = LiveHilbertMap(order=7, cell_size=4, fps=60)

    def pattern_generator():
        """Generate patterned data."""
        total = viz.grid_size**2
        for i in range(total):
            # Create attack clusters
            if 1000 <= i < 1500 or 8000 <= i < 9000:
                prediction = 1  # Attack
            else:
                prediction = 0  # Normal

            # Add some noise
            if np.random.random() < 0.05:
                prediction = 1 - prediction

            yield prediction, prediction

    viz.run(pattern_generator)


def run_with_model(model_path: str, data_path: str, order: int = 7, cell_size: int = 4):
    """
    Run live visualization with a trained model.

    Args:
        model_path: Path to model (.h5 file)
        data_path: Path to test data (y_test.csv, X_test in same dir)
        order: Hilbert curve order
        cell_size: Cell size in pixels
    """
    viz = LiveHilbertMap(order=order, cell_size=cell_size, fps=60)
    streamer = ModelStreamer(model_path, data_path)
    viz.run(lambda: streamer.stream_predictions(batch_size=32, delay=0.005))


if __name__ == "__main__":
    print("Live Hilbert Map Visualization")
    print("=" * 60)

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "random":
            demo_random_stream()
        elif mode == "pattern":
            demo_pattern_stream()
        elif mode == "model" and len(sys.argv) >= 4:
            model_path = sys.argv[2]
            data_path = sys.argv[3]
            run_with_model(model_path, data_path)
        else:
            print("Usage:")
            print("  python live_hilbert_map.py random")
            print("  python live_hilbert_map.py pattern")
            print("  python live_hilbert_map.py model <model.h5> <y_test.csv>")
    else:
        # Default: run pattern demo
        print(
            "Running pattern demo (use 'python live_hilbert_map.py random' for random)"
        )
        demo_pattern_stream()
