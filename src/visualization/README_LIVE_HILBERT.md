# Live Hilbert Map Visualization

Real-time visualization of network intrusion detection using pygame and space-filling curves.

## Features

- **Real-time visualization**: Live updates as model processes data
- **Pygame rendering**: Smooth 60 FPS display
- **Model integration**: Direct inference from trained TensorFlow/Keras models
- **Performance metrics**: FPS counter, processing rate, statistics
- **Color-coded display**: Green (normal traffic), Red (attacks)
- **Threaded processing**: Non-blocking data streaming
- **Interactive controls**: Reset, pause, and navigation

## Installation

```powershell
pip install pygame
```

## Usage

### 1. Demo Modes

**Random Stream** (simulated random traffic):
```powershell
python src\visualization\live_hilbert_map.py random
```

**Pattern Stream** (simulated attack patterns):
```powershell
python src\visualization\live_hilbert_map.py pattern
```

### 2. With Your Trained Model

```powershell
python src\visualization\live_hilbert_map.py model ^
    results/models/cnn_nsl_kdd_best.h5 ^
    data/processed/NSL_KDD/Test/y_test.csv
```

### 3. Python API

```python
from src.visualization.live_hilbert_map import LiveHilbertMap, ModelStreamer

# Option 1: Manual control
viz = LiveHilbertMap(order=7, cell_size=4, fps=60)

# Add predictions one by one
viz.add_prediction(0)  # Normal
viz.add_prediction(1)  # Attack

# Run visualization loop
viz.run()

# Option 2: With model
from src.visualization.live_hilbert_map import run_with_model

run_with_model(
    model_path='results/models/cnn_nsl_kdd_best.h5',
    data_path='data/processed/NSL_KDD/Test/y_test.csv',
    order=7,
    cell_size=4
)

# Option 3: Custom data generator
def my_generator():
    for prediction in my_predictions:
        yield prediction, None

viz = LiveHilbertMap(order=7, cell_size=4)
viz.run(my_generator)
```

## Parameters

### LiveHilbertMap

- **order** (int): Hilbert curve order, determines grid size (2^order × 2^order)
  - `order=6`: 64×64 = 4,096 points
  - `order=7`: 128×128 = 16,384 points (recommended)
  - `order=8`: 256×256 = 65,536 points
  
- **cell_size** (int): Size of each cell in pixels
  - `2-3px`: Large datasets (order 8+)
  - `4px`: Standard (order 7, recommended)
  - `6-8px`: Small datasets (order 5-6)
  
- **fps** (int): Target frames per second (default: 60)

- **title** (str): Window title

## Controls

- **R**: Reset the map
- **ESC**: Quit visualization

## Display Layout

```
┌─────────────────────┬────────────────┐
│                     │  NIDS Monitor  │
│                     │                │
│   Hilbert Map       │  Total: 1,234  │
│   (Live Updates)    │  Normal: 1,100 │
│                     │  Attacks: 134  │
│                     │                │
│                     │  Progress: 45% │
│                     │  [========>  ] │
│                     │                │
│                     │  Performance   │
│                     │  FPS: 60.0     │
│                     │  Rate: 2K/s    │
│                     │                │
│                     │  Legend        │
│                     │  ■ Unknown     │
│                     │  ■ Normal      │
│                     │  ■ Attack      │
└─────────────────────┴────────────────┘
```

## Examples

### Example 1: Quick Test
```powershell
# Pattern demo (shows attack clusters)
python src\visualization\live_hilbert_map.py pattern
```

### Example 2: CNN Model
```powershell
python src\visualization\live_hilbert_map.py model ^
    results/models/cnn_nsl_kdd_best.h5 ^
    data/processed/NSL_KDD/Test/y_test.csv
```

### Example 3: LSTM Model
```powershell
python src\visualization\live_hilbert_map.py model ^
    results/models/lstm_nsl_kdd_best.h5 ^
    data/processed/NSL_KDD/Test/y_test.csv
```

## Tips

- **Larger grids** (order 8+): Use smaller cell sizes (2-3px) for better overview
- **Smaller grids** (order 5-6): Use larger cell sizes (6-8px) for detail
- **Performance**: Higher FPS requires more CPU, lower for slower machines
- **Multiple runs**: Press 'R' to reset and reprocess data

## Technical Details

### Architecture

- **Main Thread**: Pygame rendering and event handling
- **Data Thread**: Model inference and prediction streaming
- **Queue**: Thread-safe communication between threads

### Performance

- **Rendering**: ~60 FPS on most systems
- **Processing**: Up to 2,000+ samples/second
- **Memory**: ~50-100 MB depending on grid size

### Color Scheme

- **Dark theme** for reduced eye strain
- **Green** for normal traffic (easy on eyes)
- **Red** for attacks (immediate attention)
- **Dark gray** for unprocessed areas

## Troubleshooting

**Issue**: Low FPS
- **Solution**: Reduce `fps` parameter or `cell_size`

**Issue**: Slow processing
- **Solution**: Increase batch size in ModelStreamer

**Issue**: Window too large
- **Solution**: Reduce `order` or `cell_size`

**Issue**: "TensorFlow not available"
- **Solution**: Only affects model mode, demos work without TF

## Future Enhancements

- [ ] Confidence score visualization (opacity)
- [ ] Multi-class attack types
- [ ] Zoom and pan controls
- [ ] Export video/screenshots
- [ ] Network traffic simulation mode
- [ ] Real-time network capture integration
