"""
Configuration management for pisource.

All settings can be overridden via environment variables:
  PISOURCE_HOST     - Server bind address (default: 0.0.0.0)
  PISOURCE_PORT     - Server port (default: 8000)
  PISOURCE_FPS      - Target frames per second (default: 15)
  PISOURCE_WIDTH    - Frame width in pixels (default: 640)
  PISOURCE_HEIGHT   - Frame height in pixels (default: 480)
  PISOURCE_QUALITY  - JPEG quality 1-100 (default: 80)
"""

import os


def _env_int(name: str, default: int) -> int:
    """Get an integer from environment variable with fallback."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"[config] Warning: {name}={val} is not a valid integer, using default={default}")
        return default


def _env_str(name: str, default: str) -> str:
    """Get a string from environment variable with fallback."""
    return os.environ.get(name, default)


# Server settings
HOST = _env_str("PISOURCE_HOST", "0.0.0.0")
PORT = _env_int("PISOURCE_PORT", 8000)

# Camera settings
FPS = _env_int("PISOURCE_FPS", 15)
WIDTH = _env_int("PISOURCE_WIDTH", 640)
HEIGHT = _env_int("PISOURCE_HEIGHT", 480)
JPEG_QUALITY = _env_int("PISOURCE_QUALITY", 80)

# Derived
FRAME_INTERVAL = 1.0 / FPS if FPS > 0 else 1.0 / 15


def print_config():
    """Print current configuration to stdout."""
    print("[config] Current settings:")
    print(f"  HOST       = {HOST}")
    print(f"  PORT       = {PORT}")
    print(f"  FPS        = {FPS}")
    print(f"  WIDTH      = {WIDTH}")
    print(f"  HEIGHT     = {HEIGHT}")
    print(f"  QUALITY    = {JPEG_QUALITY}")

