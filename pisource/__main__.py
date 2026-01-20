"""
Entry point for running pisource as a module.

Usage:
    python3 -m pisource [--dummy]

Options:
    --dummy    Use dummy camera for testing without hardware
"""

import sys

from .server import run_server


def main():
    """Main entry point."""
    # Simple argument parsing (avoiding argparse for minimal dependencies)
    use_dummy = "--dummy" in sys.argv or "-d" in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nEnvironment variables:")
        print("  PISOURCE_HOST     Server bind address (default: 0.0.0.0)")
        print("  PISOURCE_PORT     Server port (default: 8000)")
        print("  PISOURCE_FPS      Target FPS (default: 15)")
        print("  PISOURCE_WIDTH    Frame width (default: 640)")
        print("  PISOURCE_HEIGHT   Frame height (default: 480)")
        print("  PISOURCE_QUALITY  JPEG quality 1-100 (default: 80)")
        sys.exit(0)
    
    run_server(use_dummy_camera=use_dummy)


if __name__ == "__main__":
    main()

