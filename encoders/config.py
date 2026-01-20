"""
Configuration for encoders.

Environment variables:
  ENCODERS_DEVICE       - torch device: cuda, cpu, or auto (default: auto)
  ENCODERS_MODEL        - CLIP model name (default: ViT-B-32)
  ENCODERS_PRETRAINED   - pretrained weights (default: openai)
  ENCODERS_SNAPSHOT_INTERVAL - seconds between snapshots (default: 30)
  ENCODERS_EMBEDDING_INTERVAL - frames between embedding saves (default: 100)
  ENCODERS_ROLLING_WINDOW - size of drift history (default: 300)
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


# Device selection
DEVICE = _env_str("ENCODERS_DEVICE", "auto")

# CLIP model settings
MODEL_NAME = _env_str("ENCODERS_MODEL", "ViT-B-32")
PRETRAINED = _env_str("ENCODERS_PRETRAINED", "openai")

# Logging intervals
SNAPSHOT_INTERVAL = _env_float("ENCODERS_SNAPSHOT_INTERVAL", 30.0)  # seconds
EMBEDDING_SAVE_INTERVAL = _env_int("ENCODERS_EMBEDDING_INTERVAL", 100)  # frames

# Metrics
ROLLING_WINDOW_SIZE = _env_int("ENCODERS_ROLLING_WINDOW", 300)

# Stream settings
RECONNECT_DELAY = _env_float("ENCODERS_RECONNECT_DELAY", 2.0)  # seconds
FRAME_TIMEOUT = _env_float("ENCODERS_FRAME_TIMEOUT", 10.0)  # seconds


def get_device() -> str:
    """Resolve device setting to actual torch device string."""
    if DEVICE == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return DEVICE


def print_config():
    """Print current configuration."""
    print("[config] Settings:")
    print(f"  DEVICE (resolved)    = {get_device()}")
    print(f"  MODEL                = {MODEL_NAME}")
    print(f"  PRETRAINED           = {PRETRAINED}")
    print(f"  SNAPSHOT_INTERVAL    = {SNAPSHOT_INTERVAL}s")
    print(f"  EMBEDDING_INTERVAL   = {EMBEDDING_SAVE_INTERVAL} frames")
    print(f"  ROLLING_WINDOW       = {ROLLING_WINDOW_SIZE}")

