"""
Logging utilities for saving embeddings, metrics, and snapshots.

Creates run directories with:
- config.json: Run configuration
- drift.csv: Timestamp + cosine similarity log
- embeddings/: Saved embedding arrays
- snapshots/: Periodic frame captures
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from . import config


class RunLogger:
    """
    Manages logging for a single run session.
    
    Creates structured output directory:
        runs/
          YYYYMMDD_HHMMSS/
            config.json
            drift.csv
            embeddings/
              batch_0000.npz
              batch_0001.npz
              ...
            snapshots/
              frame_000000.jpg
              frame_000100.jpg
              ...
    
    Usage:
        logger = RunLogger(stream_url="http://...")
        logger.start()
        
        logger.log_drift(timestamp, similarity)
        logger.save_embeddings(embeddings, timestamps, indices)
        logger.save_snapshot(frame, frame_idx)
        
        logger.stop()
    """
    
    def __init__(
        self,
        stream_url: str,
        base_dir: str = "runs",
        run_name: str = None
    ):
        self.stream_url = stream_url
        self.base_dir = Path(base_dir)
        
        # Create run name from timestamp if not provided
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        
        self.run_dir: Optional[Path] = None
        self.embeddings_dir: Optional[Path] = None
        self.snapshots_dir: Optional[Path] = None
        
        self._drift_file = None
        self._drift_writer = None
        self._embedding_batch = 0
        self._snapshot_count = 0
        self._last_snapshot_time = 0.0
        self._started = False
    
    def start(self):
        """Create run directory and initialize logging."""
        if self._started:
            return
        
        # Create directories
        self.run_dir = self.base_dir / self.run_name
        self.embeddings_dir = self.run_dir / "embeddings"
        self.snapshots_dir = self.run_dir / "snapshots"
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        print(f"[logger] Created run directory: {self.run_dir}")
        
        # Save config
        self._save_config()
        
        # Initialize drift CSV
        drift_path = self.run_dir / "drift.csv"
        self._drift_file = open(drift_path, 'w', newline='')
        self._drift_writer = csv.writer(self._drift_file)
        self._drift_writer.writerow(['timestamp', 'frame_idx', 'similarity', 'drift'])
        
        self._started = True
        self._last_snapshot_time = time.time()
    
    def _save_config(self):
        """Save run configuration to JSON."""
        cfg = {
            "stream_url": self.stream_url,
            "run_name": self.run_name,
            "start_time": datetime.now().isoformat(),
            "settings": {
                "device": config.get_device(),
                "model": config.MODEL_NAME,
                "pretrained": config.PRETRAINED,
                "snapshot_interval": config.SNAPSHOT_INTERVAL,
                "embedding_save_interval": config.EMBEDDING_SAVE_INTERVAL,
                "rolling_window": config.ROLLING_WINDOW_SIZE
            }
        }
        
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
    
    def log_drift(self, timestamp: float, frame_idx: int, similarity: float):
        """
        Log a single drift measurement.
        
        Args:
            timestamp: Unix timestamp
            frame_idx: Frame number
            similarity: Cosine similarity value
        """
        if not self._started or self._drift_writer is None:
            return
        
        drift = 1.0 - similarity
        self._drift_writer.writerow([
            f"{timestamp:.6f}",
            frame_idx,
            f"{similarity:.6f}",
            f"{drift:.6f}"
        ])
        
        # Flush periodically
        if frame_idx % 100 == 0:
            self._drift_file.flush()
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        timestamps: np.ndarray,
        frame_indices: np.ndarray
    ):
        """
        Save a batch of embeddings to npz file.
        
        Args:
            embeddings: Array of shape (N, D)
            timestamps: Array of shape (N,)
            frame_indices: Array of shape (N,)
        """
        if not self._started:
            return
        
        if len(embeddings) == 0:
            return
        
        filename = f"batch_{self._embedding_batch:04d}.npz"
        filepath = self.embeddings_dir / filename
        
        np.savez_compressed(
            filepath,
            embeddings=embeddings,
            timestamps=timestamps,
            frame_indices=frame_indices
        )
        
        print(f"[logger] Saved {len(embeddings)} embeddings to {filename}")
        self._embedding_batch += 1
    
    def save_snapshot(self, frame: np.ndarray, frame_idx: int, force: bool = False):
        """
        Save frame snapshot if enough time has passed.
        
        Args:
            frame: BGR numpy array
            frame_idx: Frame number
            force: Save regardless of time interval
        """
        if not self._started:
            return
        
        now = time.time()
        elapsed = now - self._last_snapshot_time
        
        if not force and elapsed < config.SNAPSHOT_INTERVAL:
            return
        
        filename = f"frame_{frame_idx:06d}.jpg"
        filepath = self.snapshots_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        
        self._last_snapshot_time = now
        self._snapshot_count += 1
        print(f"[logger] Saved snapshot: {filename}")
    
    def stop(self):
        """Finalize logging and close files."""
        if not self._started:
            return
        
        print(f"[logger] Finalizing run: {self.run_name}")
        
        # Close drift file
        if self._drift_file:
            self._drift_file.close()
            self._drift_file = None
        
        # Save summary
        summary = {
            "run_name": self.run_name,
            "end_time": datetime.now().isoformat(),
            "embedding_batches": self._embedding_batch,
            "snapshots": self._snapshot_count
        }
        
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[logger] Run complete. Data saved to: {self.run_dir}")
        self._started = False
    
    def get_run_dir(self) -> Optional[Path]:
        """Get the run directory path."""
        return self.run_dir


def load_run_embeddings(run_dir: str) -> tuple:
    """
    Load all embeddings from a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        (embeddings, timestamps, frame_indices) concatenated arrays
    """
    run_path = Path(run_dir)
    embeddings_dir = run_path / "embeddings"
    
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"No embeddings directory in {run_dir}")
    
    all_embeddings = []
    all_timestamps = []
    all_indices = []
    
    for npz_file in sorted(embeddings_dir.glob("batch_*.npz")):
        data = np.load(npz_file)
        all_embeddings.append(data['embeddings'])
        all_timestamps.append(data['timestamps'])
        all_indices.append(data['frame_indices'])
    
    if not all_embeddings:
        return np.array([]), np.array([]), np.array([])
    
    return (
        np.concatenate(all_embeddings),
        np.concatenate(all_timestamps),
        np.concatenate(all_indices)
    )

