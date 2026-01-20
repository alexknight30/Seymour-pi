# encoders - Vision Stream Encoder

Receives MJPEG stream from Raspberry Pi, produces CLIP embeddings, and tracks visual drift over time.

## Features

- Connects to MJPEG HTTP stream from `pisource`
- Generates CLIP embeddings (ViT-B/32) for each frame
- Tracks cosine similarity drift between consecutive frames
- Live video display with metrics overlay
- Logs embeddings, drift values, and snapshots

## Requirements

- Windows/Mac/Linux with Python 3.10+
- GPU recommended for real-time performance (CUDA)
- Network access to Raspberry Pi

## Installation (Windows)

### 1. Create Virtual Environment

```powershell
cd Seymour-pi\encoders

# Create venv
python -m venv .venv

# Activate
.venv\Scripts\activate
```

### 2. Install PyTorch

For **GPU (NVIDIA CUDA)**:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For **CPU only**:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Other Dependencies

```powershell
pip install -r requirements.txt
```

## Usage

### Basic Run

```powershell
# Make sure Pi stream is running first!
python -m encoders --stream http://<PI_IP>:8000/stream.mjpg
```

Replace `<PI_IP>` with your Raspberry Pi's IP address (e.g., `192.168.1.100`).

### Command Line Options

| Option | Description |
|--------|-------------|
| `--stream URL` | MJPEG stream URL (required) |
| `--no-display` | Run without video window (headless) |
| `--no-plot` | Disable matplotlib drift plot |
| `--no-save` | Disable logging (no files saved) |
| `--test` | Test connection only, then exit |

### Examples

```powershell
# Test connection
python -m encoders --stream http://192.168.1.100:8000/stream.mjpg --test

# Run without saving logs
python -m encoders --stream http://192.168.1.100:8000/stream.mjpg --no-save

# Headless mode (no display)
python -m encoders --stream http://192.168.1.100:8000/stream.mjpg --no-display
```

### Keyboard Controls

- **Q** or **ESC**: Quit
- **Ctrl+C**: Interrupt

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENCODERS_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `ENCODERS_MODEL` | `ViT-B-32` | CLIP model name |
| `ENCODERS_PRETRAINED` | `openai` | Pretrained weights |
| `ENCODERS_SNAPSHOT_INTERVAL` | `30` | Seconds between snapshots |
| `ENCODERS_EMBEDDING_INTERVAL` | `100` | Frames between embedding saves |
| `ENCODERS_ROLLING_WINDOW` | `300` | Drift history size |

Example:

```powershell
$env:ENCODERS_DEVICE="cuda"
python -m encoders --stream http://192.168.1.100:8000/stream.mjpg
```

## Output

Runs are saved to `runs/YYYYMMDD_HHMMSS/`:

```
runs/
  20260107_143022/
    config.json       # Run configuration
    drift.csv         # Timestamp, frame, similarity, drift
    summary.json      # Final summary stats
    embeddings/
      batch_0000.npz  # Embedding arrays
      batch_0001.npz
      ...
    snapshots/
      frame_000000.jpg
      frame_000030.jpg
      ...
```

### Loading Saved Embeddings

```python
from encoders.logging_utils import load_run_embeddings

embeddings, timestamps, indices = load_run_embeddings("runs/20260107_143022")
print(f"Loaded {len(embeddings)} embeddings of shape {embeddings.shape}")
```

## Display

The live display shows:

- **Video feed**: Real-time camera stream
- **FPS**: Processing frame rate
- **Similarity**: Cosine similarity with previous frame (1.0 = identical)
- **Drift**: 1 - similarity (0.0 = no change)
- **Drift bar**: Visual indicator of scene change intensity

## Troubleshooting

### "Connection refused" or timeout

1. Check Pi is running: `curl http://<PI_IP>:8000/health`
2. Check firewall allows port 8000
3. Verify both devices on same network
4. Try using Pi's IP instead of hostname

### Slow / low FPS

1. Use GPU if available (check `ENCODERS_DEVICE`)
2. Reduce Pi stream resolution
3. Close other GPU-intensive applications

### "CUDA out of memory"

1. Reduce batch processing (currently single-frame)
2. Use smaller model (not currently supported, would need code changes)
3. Use CPU: `$env:ENCODERS_DEVICE="cpu"`

### OpenCV window not appearing

1. Make sure OpenCV is installed correctly
2. Try running without display: `--no-display`
3. On headless systems, use X forwarding or `--no-display`

### Import errors

Make sure you activated the venv:
```powershell
.venv\Scripts\activate
```

## Architecture

```
__main__.py       Entry point, orchestrates pipeline
receiver.py       MJPEG stream decoder
clip_encoder.py   CLIP model wrapper
metrics.py        Cosine similarity + drift tracking
viz.py            OpenCV display + matplotlib plot
logging_utils.py  File output (csv, npz, jpg)
config.py         Environment configuration
```

## Performance Notes

- **GPU (RTX 3060)**: ~30 FPS
- **GPU (GTX 1060)**: ~15 FPS
- **CPU (modern i7)**: ~3-5 FPS
- **CPU (older)**: ~1-2 FPS

CLIP encoding is the bottleneck. The MJPEG decode is very fast.

## License

MIT

