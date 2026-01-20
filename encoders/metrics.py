"""
Metrics computation for embedding analysis.

Tracks cosine similarity between consecutive embeddings
and maintains rolling statistics.
"""

from collections import deque
from typing import Optional, Tuple
import numpy as np

from . import config


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a, b: 1D numpy arrays (should be normalized for best results)
        
    Returns:
        Cosine similarity in range [-1, 1]
    """
    if a is None or b is None:
        return 0.0
    
    # Compute dot product
    dot = np.dot(a, b)
    
    # Compute norms (in case not pre-normalized)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot / (norm_a * norm_b))


class DriftTracker:
    """
    Tracks embedding drift (cosine similarity) over time.
    
    Maintains a rolling window of similarity values and computes statistics.
    
    Usage:
        tracker = DriftTracker()
        
        for embedding in embeddings:
            similarity = tracker.update(embedding)
            print(f"Drift: {1 - similarity:.4f}")
        
        stats = tracker.get_stats()
    """
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size or config.ROLLING_WINDOW_SIZE
        
        self._history: deque = deque(maxlen=self.window_size)
        self._prev_embedding: Optional[np.ndarray] = None
        self._count = 0
    
    def update(self, embedding: np.ndarray) -> float:
        """
        Update tracker with new embedding.
        
        Args:
            embedding: New embedding vector
            
        Returns:
            Cosine similarity with previous embedding (1.0 for first frame)
        """
        if embedding is None:
            return 1.0
        
        if self._prev_embedding is None:
            # First embedding
            self._prev_embedding = embedding.copy()
            self._count = 1
            return 1.0
        
        # Compute similarity
        similarity = cosine_similarity(embedding, self._prev_embedding)
        
        # Store in history
        self._history.append(similarity)
        
        # Update state
        self._prev_embedding = embedding.copy()
        self._count += 1
        
        return similarity
    
    def get_history(self) -> np.ndarray:
        """Get similarity history as numpy array."""
        return np.array(self._history)
    
    def get_stats(self) -> dict:
        """
        Get statistics from the rolling window.
        
        Returns:
            Dict with mean, std, min, max of similarities
        """
        if not self._history:
            return {
                "count": 0,
                "mean": 1.0,
                "std": 0.0,
                "min": 1.0,
                "max": 1.0
            }
        
        arr = np.array(self._history)
        
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
    
    def get_latest(self) -> Tuple[float, float]:
        """
        Get latest similarity and drift.
        
        Returns:
            (similarity, drift) where drift = 1 - similarity
        """
        if not self._history:
            return 1.0, 0.0
        
        sim = self._history[-1]
        return sim, 1.0 - sim
    
    @property
    def total_count(self) -> int:
        """Total number of embeddings processed."""
        return self._count
    
    def reset(self):
        """Reset tracker state."""
        self._history.clear()
        self._prev_embedding = None
        self._count = 0


class EmbeddingBuffer:
    """
    Buffer for collecting embeddings before saving.
    
    Stores embeddings with timestamps for periodic saving to disk.
    """
    
    def __init__(self, capacity: int = None):
        self.capacity = capacity or config.EMBEDDING_SAVE_INTERVAL
        
        self._embeddings: list = []
        self._timestamps: list = []
        self._frame_indices: list = []
    
    def add(self, embedding: np.ndarray, timestamp: float, frame_idx: int):
        """Add an embedding to the buffer."""
        self._embeddings.append(embedding.copy())
        self._timestamps.append(timestamp)
        self._frame_indices.append(frame_idx)
    
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return len(self._embeddings) >= self.capacity
    
    def get_and_clear(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all data and clear buffer.
        
        Returns:
            (embeddings, timestamps, frame_indices) as numpy arrays
        """
        if not self._embeddings:
            return np.array([]), np.array([]), np.array([])
        
        embeddings = np.stack(self._embeddings)
        timestamps = np.array(self._timestamps)
        indices = np.array(self._frame_indices)
        
        # Clear
        self._embeddings = []
        self._timestamps = []
        self._frame_indices = []
        
        return embeddings, timestamps, indices
    
    def __len__(self) -> int:
        return len(self._embeddings)

