"""
Entry point for the encoders module.

Usage:
    python -m encoders --stream http://<PI_IP>:8000/stream.mjpg

Options:
    --stream URL    MJPEG stream URL (required)
    --no-display    Run without video display window
    --no-plot       Disable matplotlib drift plot
    --no-save       Disable logging (no files saved)
    --test          Test connection only, then exit
"""

import argparse
import sys
import time

from . import config
from .receiver import MJPEGReceiver, test_connection
from .clip_encoder import CLIPEncoder
from .metrics import DriftTracker, EmbeddingBuffer
from .viz import LiveDisplay, simple_drift_meter
from .logging_utils import RunLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision stream encoder - connects to Pi MJPEG stream and produces CLIP embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m encoders --stream http://192.168.1.100:8000/stream.mjpg
    python -m encoders --stream http://pi.local:8000/stream.mjpg --no-plot
    python -m encoders --stream http://192.168.1.100:8000/stream.mjpg --test
        """
    )
    
    parser.add_argument(
        "--stream", "-s",
        required=True,
        help="MJPEG stream URL (e.g., http://192.168.1.100:8000/stream.mjpg)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without video display (headless mode)"
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib drift plot"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable logging (no files saved)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connection only, then exit"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("  SEYMOUR VISION ENCODER")
    print("=" * 60)
    print()
    
    config.print_config()
    print()
    print(f"[main] Stream URL: {args.stream}")
    print()
    
    # Test mode - just check connection
    if args.test:
        print("[main] Testing connection...")
        if test_connection(args.stream):
            print("[main] Connection successful!")
            return 0
        else:
            print("[main] Connection failed!")
            return 1
    
    # Initialize components
    print("[main] Initializing components...")
    
    # 1. CLIP encoder
    try:
        encoder = CLIPEncoder()
        encoder.load()
    except Exception as e:
        print(f"[main] Failed to load CLIP encoder: {e}")
        return 1
    
    # 2. Stream receiver
    receiver = MJPEGReceiver(args.stream)
    
    # 3. Metrics tracker
    tracker = DriftTracker()
    embedding_buffer = EmbeddingBuffer()
    
    # 4. Display (optional)
    display = None
    if not args.no_display:
        display = LiveDisplay()
        display.start()
    
    # 5. Logger (optional)
    logger = None
    if not args.no_save:
        logger = RunLogger(stream_url=args.stream)
        logger.start()
    
    print()
    print("[main] Starting processing loop...")
    print("[main] Press 'q' in video window or Ctrl+C to stop")
    print()
    
    frame_idx = 0
    start_time = time.time()
    
    try:
        for frame in receiver.frames():
            timestamp = time.time()
            
            # Encode frame
            embedding = encoder.encode(frame)
            
            if embedding is None:
                print(f"[main] Frame {frame_idx}: encode failed, skipping")
                frame_idx += 1
                continue
            
            # Track drift
            similarity = tracker.update(embedding)
            _, drift = tracker.get_latest()
            
            # Log metrics
            if logger:
                logger.log_drift(timestamp, frame_idx, similarity)
                logger.save_snapshot(frame, frame_idx)
            
            # Buffer embeddings
            embedding_buffer.add(embedding, timestamp, frame_idx)
            
            # Save embeddings periodically
            if embedding_buffer.is_full() and logger:
                emb, ts, idx = embedding_buffer.get_and_clear()
                logger.save_embeddings(emb, ts, idx)
            
            # Update display
            if display:
                elapsed = timestamp - start_time
                extra = f"Elapsed: {elapsed:.1f}s | Frames: {frame_idx}"
                
                if not display.update(frame, similarity, extra):
                    print("[main] Display closed")
                    break
            else:
                # Print text-based meter to console
                if frame_idx % 10 == 0:  # Every 10 frames
                    meter = simple_drift_meter(similarity)
                    print(f"Frame {frame_idx:6d}: {meter}")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user")
    
    except Exception as e:
        print(f"[main] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print()
        print("[main] Shutting down...")
        
        # Stop receiver
        receiver.stop()
        
        # Save remaining embeddings
        if logger and len(embedding_buffer) > 0:
            emb, ts, idx = embedding_buffer.get_and_clear()
            logger.save_embeddings(emb, ts, idx)
        
        # Close display
        if display:
            display.stop()
        
        # Finalize logger
        if logger:
            logger.stop()
        
        # Print summary
        elapsed = time.time() - start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        
        print()
        print("[main] Session Summary:")
        print(f"  Total frames: {frame_idx}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {fps:.1f}")
        
        stats = tracker.get_stats()
        print(f"  Drift stats:")
        print(f"    Mean similarity: {stats['mean']:.4f}")
        print(f"    Std deviation: {stats['std']:.4f}")
        print(f"    Min similarity: {stats['min']:.4f}")
        print(f"    Max similarity: {stats['max']:.4f}")
        
        if logger:
            print(f"  Output saved to: {logger.get_run_dir()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

