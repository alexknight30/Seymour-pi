"""
Entry point for the language module.

Can run standalone for testing, or integrate with the encoder pipeline.

Usage:
    # Test mode - process saved embeddings
    python -m language --test
    
    # Live mode - connect to encoder output
    python -m language --stream http://<PI_IP>:8000/stream.mjpg
    
    # Interactive labeling mode
    python -m language --interactive --embeddings path/to/embeddings.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .projection import ProjectionLayer, FeedbackBuffer
from .llm import create_llm
from .feedback import FeedbackLoop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Language module - LLM integration for visual understanding"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with synthetic embeddings"
    )
    
    parser.add_argument(
        "--stream",
        type=str,
        help="MJPEG stream URL for live processing"
    )
    
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to saved embeddings (.npz) for batch processing"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - manually label embeddings"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "local"],
        help="LLM provider override"
    )
    
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Disable automatic training"
    )
    
    return parser.parse_args()


def test_mode():
    """Test the feedback loop with synthetic embeddings."""
    print("=" * 50)
    print("  LANGUAGE MODULE - TEST MODE")
    print("=" * 50)
    print()
    
    config.print_config()
    print()
    
    # Create feedback loop
    loop = FeedbackLoop(auto_train=True, train_interval=5)
    
    try:
        loop.start()
    except Exception as e:
        print(f"[test] Could not start loop: {e}")
        print("[test] Continuing without LLM...")
        loop._running = True
    
    print()
    print("[test] Generating synthetic embeddings...")
    print()
    
    # Generate some test embeddings
    for i in range(20):
        # Create synthetic embedding (random but consistent for same "scene")
        scene_id = i % 5  # 5 different "scenes"
        np.random.seed(scene_id)
        embedding = np.random.randn(config.CLIP_EMBEDDING_DIM).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        # Process
        label, projected = loop.process(
            embedding,
            context=f"Frame {i}, scene {scene_id}",
            get_label=(loop.llm is not None)
        )
        
        if label:
            print(f"Frame {i:3d} | Scene {scene_id} | Label: {label}")
        else:
            print(f"Frame {i:3d} | Scene {scene_id} | Projected dim: {len(projected)}")
        
        time.sleep(0.1)  # Simulate frame rate
    
    print()
    print("[test] Stats:", loop.get_stats())
    
    # Save state
    loop.stop()
    
    print()
    print("[test] Done!")


def live_mode(stream_url: str, auto_train: bool = True):
    """Process live stream with LLM feedback."""
    print("=" * 50)
    print("  LANGUAGE MODULE - LIVE MODE")
    print("=" * 50)
    print()
    
    # Import encoder components
    try:
        from encoders.receiver import MJPEGReceiver
        from encoders.clip_encoder import CLIPEncoder
    except ImportError:
        print("[live] Error: Could not import encoder modules.")
        print("[live] Make sure you're running from the project root.")
        return 1
    
    config.print_config()
    print()
    
    # Initialize components
    print("[live] Loading CLIP encoder...")
    encoder = CLIPEncoder()
    encoder.load()
    
    print("[live] Connecting to stream...")
    receiver = MJPEGReceiver(stream_url)
    
    print("[live] Starting feedback loop...")
    loop = FeedbackLoop(auto_train=auto_train, train_interval=30)
    loop.start()
    
    print()
    print("[live] Processing... Press Ctrl+C to stop")
    print()
    
    frame_count = 0
    
    try:
        for frame in receiver.frames():
            # Get CLIP embedding
            embedding = encoder.encode(frame)
            
            if embedding is None:
                continue
            
            # Process through feedback loop (only get label every 30 frames)
            get_label = (frame_count % 30 == 0) and (loop.llm is not None)
            
            label, projected = loop.process(
                embedding,
                context=f"Frame {frame_count}",
                get_label=get_label
            )
            
            if label:
                print(f"Frame {frame_count:5d} | Label: {label}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n[live] Interrupted")
    
    finally:
        receiver.stop()
        loop.stop()
    
    print()
    print("[live] Stats:", loop.get_stats())
    return 0


def interactive_mode(embeddings_path: str):
    """Manually label embeddings for training."""
    print("=" * 50)
    print("  LANGUAGE MODULE - INTERACTIVE LABELING")
    print("=" * 50)
    print()
    
    # Load embeddings
    path = Path(embeddings_path)
    if not path.exists():
        print(f"[interactive] File not found: {embeddings_path}")
        return 1
    
    data = np.load(path)
    embeddings = data['embeddings']
    
    print(f"[interactive] Loaded {len(embeddings)} embeddings")
    print()
    
    # Create feedback loop (no LLM needed)
    loop = FeedbackLoop(auto_train=False)
    loop._running = True
    loop.projection.load()
    
    print("Instructions:")
    print("  - For each embedding, enter a short label (1-5 words)")
    print("  - Press Enter to skip")
    print("  - Type 'quit' to stop and save")
    print("  - Type 'train' to run a training step")
    print()
    
    labeled = 0
    
    try:
        for i, embedding in enumerate(embeddings):
            # Show embedding info
            print(f"\nEmbedding {i+1}/{len(embeddings)}")
            print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
            
            label = input("Label: ").strip()
            
            if label.lower() == 'quit':
                break
            elif label.lower() == 'train':
                loss = loop.train_step()
                print(f"  Training loss: {loss:.4f}")
                continue
            elif label:
                loop.add_manual_label(embedding, label)
                labeled += 1
    
    except KeyboardInterrupt:
        print("\n[interactive] Interrupted")
    
    print()
    print(f"[interactive] Labeled {labeled} embeddings")
    print(f"[interactive] Buffer size: {len(loop.feedback_buffer)}")
    
    # Final training
    if labeled > 0:
        print("[interactive] Running final training...")
        for _ in range(5):
            loss = loop.train_step()
            print(f"  Loss: {loss:.4f}")
    
    # Save
    loop.projection.save()
    Path("data").mkdir(exist_ok=True)
    loop.feedback_buffer.save("data/feedback_buffer.npz")
    
    print("[interactive] Done!")
    return 0


def main():
    args = parse_args()
    
    # Override provider if specified
    if args.provider:
        config.LLM_PROVIDER = args.provider
    
    if args.test:
        return test_mode()
    elif args.stream:
        return live_mode(args.stream, auto_train=not args.no_train)
    elif args.interactive and args.embeddings:
        return interactive_mode(args.embeddings)
    else:
        print("Usage:")
        print("  python -m language --test")
        print("  python -m language --stream http://<PI_IP>:8000/stream.mjpg")
        print("  python -m language --interactive --embeddings path/to/embeddings.npz")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)

