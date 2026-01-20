"""
Configuration for language module.

Environment variables:
  LANGUAGE_LLM_PROVIDER   - LLM provider: openai, anthropic, local (default: local)
  LANGUAGE_LLM_MODEL      - Model name (default: depends on provider)
  LANGUAGE_DEVICE         - torch device: cuda, cpu, auto (default: auto)
  LANGUAGE_EMBEDDING_DIM  - CLIP embedding dimension (default: 512)
  LANGUAGE_PROJECTION_DIM - Projection output dimension (default: 768)
"""

import os


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"[config] Warning: {name}={val} invalid, using default={default}")
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"[config] Warning: {name}={val} invalid, using default={default}")
        return default


# LLM settings
LLM_PROVIDER = _env_str("LANGUAGE_LLM_PROVIDER", "local")  # openai, anthropic, local
LLM_MODEL = _env_str("LANGUAGE_LLM_MODEL", "")  # Empty = use provider default

# API keys (for cloud providers)
OPENAI_API_KEY = _env_str("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = _env_str("ANTHROPIC_API_KEY", "")

# Model dimensions
CLIP_EMBEDDING_DIM = _env_int("LANGUAGE_EMBEDDING_DIM", 512)  # CLIP ViT-B/32 output
PROJECTION_DIM = _env_int("LANGUAGE_PROJECTION_DIM", 768)  # Common LLM hidden size

# Training settings
LEARNING_RATE = _env_float("LANGUAGE_LEARNING_RATE", 1e-4)
FEEDBACK_BATCH_SIZE = _env_int("LANGUAGE_FEEDBACK_BATCH_SIZE", 32)

# Device
DEVICE = _env_str("LANGUAGE_DEVICE", "auto")


def get_device() -> str:
    """Resolve device setting to actual torch device string."""
    if DEVICE == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return DEVICE


def get_llm_model() -> str:
    """Get the LLM model name based on provider."""
    if LLM_MODEL:
        return LLM_MODEL
    
    # Provider defaults
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "local": "llama2"  # For ollama
    }
    return defaults.get(LLM_PROVIDER, "gpt-4o-mini")


def print_config():
    """Print current configuration."""
    print("[language config] Settings:")
    print(f"  LLM_PROVIDER     = {LLM_PROVIDER}")
    print(f"  LLM_MODEL        = {get_llm_model()}")
    print(f"  DEVICE           = {get_device()}")
    print(f"  CLIP_DIM         = {CLIP_EMBEDDING_DIM}")
    print(f"  PROJECTION_DIM   = {PROJECTION_DIM}")

