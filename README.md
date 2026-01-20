# Seymour-pi

Vision encoding pipeline: Raspberry Pi camera stream to laptop embeddings to LLM understanding.

This project explores Quine's problem of language through real-time visual stream encoding and grounded language learning.

## Architecture

```
┌─────────────────┐      ┌─────────────────────────────────────────────────┐
│  RASPBERRY PI   │      │                    LAPTOP                       │
│                 │      │                                                 │
│  Camera Module  │      │  ┌─────────┐    ┌──────────┐    ┌───────────┐  │
│       │         │ WiFi │  │Receiver │    │  CLIP    │    │Projection │  │
│       v         │ ───> │  │ (MJPEG) │ -> │ Encoder  │ -> │  Layer    │  │
│  MJPEG Stream   │:8000 │  └─────────┘    └──────────┘    └───────────┘  │
│                 │      │                      │               │         │
└─────────────────┘      │                      v               v         │
     pisource/           │                 Embeddings    ┌───────────┐    │
                         │                      │        │    LLM    │    │
                         │                      v        │  (Label)  │    │
                         │              Drift Metrics    └───────────┘    │
                         │                                     │          │
                         │                    ┌────────────────┘          │
                         │                    v                           │
                         │            Feedback Loop                       │
                         │         (trains projection)                    │
                         └─────────────────────────────────────────────────┘
                               encoders/              language/
```

## Components

| Folder | Runs On | Purpose |
|--------|---------|---------|
| `pisource/` | Raspberry Pi | MJPEG camera stream server |
| `encoders/` | Laptop | Stream receiver + CLIP embeddings |
| `language/` | Laptop | LLM integration + feedback loop |

## The Pipeline

1. **Capture** (Pi): Camera captures frames, streams as MJPEG over HTTP
2. **Encode** (Laptop): CLIP encodes frames into 512-dim embeddings
3. **Project** (Laptop): Learned projection maps to LLM-compatible space
4. **Label** (Laptop): LLM generates semantic labels for visual input
5. **Learn** (Laptop): Projection layer trains on LLM feedback

## Quick Start

### 1. Raspberry Pi Setup

```bash
# On the Raspberry Pi
cd Seymour-pi/pisource
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pisource
```

Find Pi IP: `hostname -I`

### 2. Laptop Encoder Setup (Windows)

```powershell
cd Seymour-pi\encoders
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch (GPU recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
python -m encoders --stream http://<PI_IP>:8000/stream.mjpg
```

### 3. Language/LLM Setup (Optional)

```powershell
cd Seymour-pi\language
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Set your LLM provider
$env:OPENAI_API_KEY="sk-your-key"
$env:LANGUAGE_LLM_PROVIDER="openai"

python -m language --stream http://<PI_IP>:8000/stream.mjpg
```

## Expected Output

- Live video window showing Pi camera feed
- Real-time cosine similarity drift (how much the scene is changing)
- LLM-generated labels for visual scenes
- Logs in `runs/YYYYMMDD_HHMMSS/`:
  - `drift.csv` - timestamp + cosine similarity
  - `embeddings.npz` - CLIP embedding vectors
  - `snapshots/` - periodic frame captures
- Learned projection weights in `models/projection.pt`

## Documentation

- [pisource/README.md](pisource/README.md) - Pi camera streaming
- [encoders/README.md](encoders/README.md) - CLIP encoder + metrics
- [language/README.md](language/README.md) - LLM integration + feedback loop

## Requirements

- Raspberry Pi 4 with Camera Module 3
- Laptop with Python 3.10+ (GPU recommended)
- Both devices on same network
- (Optional) OpenAI/Anthropic API key or local Ollama

## Quine's Problem

This project explores how meaning can be grounded in perceptual experience. The feedback loop between visual encoding and language creates a system where:

- **Visual experience** provides the raw input
- **CLIP embeddings** capture semantic features
- **LLM labels** provide linguistic interpretation
- **The projection layer learns** to bridge perception and language

Over time, the system develops its own grounded understanding of the visual world.
