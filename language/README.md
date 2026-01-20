# language - LLM Integration for Visual Understanding

This module connects the visual encoder to language models, enabling:
- Semantic labeling of visual embeddings
- Feedback loop for encoder refinement
- Grounding language in visual experience

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    CLIP     │     │ Projection  │     │     LLM     │
│   Encoder   │ --> │   Layer     │ --> │   (Label)   │
│             │     │ (learnable) │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
       ^                   ^                   │
       │                   │                   │
       │                   └───────────────────┘
       │                     Training signal
       │
       └── (Optional) Fine-tune encoder
```

## The Feedback Loop

1. **Encode**: CLIP converts image to 512-dim embedding
2. **Project**: Learned projection layer maps to LLM-compatible space
3. **Label**: LLM generates description/label for the visual input
4. **Learn**: Projection layer trains to align visual embeddings with text embeddings of labels
5. **Repeat**: Over time, the projection learns better visual-language alignment

## Installation

```powershell
cd Seymour-pi\language
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch (see encoders README for CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## LLM Providers

### Option 1: OpenAI (Recommended for quality)

```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
$env:LANGUAGE_LLM_PROVIDER="openai"
python -m language --test
```

### Option 2: Anthropic Claude

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
$env:LANGUAGE_LLM_PROVIDER="anthropic"
python -m language --test
```

### Option 3: Local with Ollama (Free, private)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Run:

```powershell
$env:LANGUAGE_LLM_PROVIDER="local"
$env:LANGUAGE_LLM_MODEL="llama2"
python -m language --test
```

## Usage

### Test Mode

```powershell
python -m language --test
```

Generates synthetic embeddings and tests the feedback loop.

### Live Mode (with Pi stream)

```powershell
python -m language --stream http://<PI_IP>:8000/stream.mjpg
```

Processes live camera feed with LLM labeling and trains projection layer.

### Interactive Labeling

Manually label embeddings for supervised training:

```powershell
python -m language --interactive --embeddings runs/YYYYMMDD_HHMMSS/embeddings/batch_0000.npz
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGUAGE_LLM_PROVIDER` | `local` | `openai`, `anthropic`, or `local` |
| `LANGUAGE_LLM_MODEL` | auto | Model name (provider-specific) |
| `LANGUAGE_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `LANGUAGE_EMBEDDING_DIM` | `512` | CLIP embedding dimension |
| `LANGUAGE_PROJECTION_DIM` | `768` | Projection output dimension |
| `LANGUAGE_LEARNING_RATE` | `1e-4` | Projection training LR |

## Files Created

```
models/
  projection.pt       # Learned projection weights

data/
  feedback_buffer.npz # Accumulated (embedding, label) pairs
```

## How the Learning Works

### Projection Layer Training

The projection layer is a small MLP:
```
CLIP (512) -> Linear -> ReLU -> LayerNorm -> Linear -> LayerNorm -> Output (768)
```

It's trained with cosine similarity loss:
- Input: CLIP visual embedding
- Target: Text embedding of LLM's label
- Goal: Make projected visual embedding similar to text embedding

### Why This Works

CLIP was trained to align images and text in a shared embedding space. By training a projection layer with LLM feedback, we:

1. **Adapt** to your specific visual domain (what your camera sees)
2. **Refine** the semantic alignment based on LLM's understanding
3. **Learn** which visual features matter for your use case

## Integration with Encoders

The language module can run standalone or integrate with the encoder pipeline:

```python
from encoders.clip_encoder import CLIPEncoder
from language.feedback import FeedbackLoop

encoder = CLIPEncoder()
encoder.load()

loop = FeedbackLoop()
loop.start()

# For each frame
embedding = encoder.encode(frame)
label, projected = loop.process(embedding)
print(f"Scene: {label}")
```

## Future: Encoder Fine-tuning

The `EncoderFineTuner` class (in `feedback.py`) provides scaffolding for fine-tuning the CLIP encoder itself based on accumulated feedback. This is more compute-intensive but can improve the base representations.

Currently not fully implemented - use projection layer training instead.

## Quine's Problem

This architecture relates to Quine's problem of radical translation and the indeterminacy of meaning. By grounding LLM labels in visual experience, we create a feedback loop where:

- Visual experience provides the "stimulus meaning"
- Language (LLM labels) provides the symbolic representation
- The projection layer learns the mapping between them

Over time, this creates a system where meaning is grounded in perception rather than purely symbolic manipulation.

