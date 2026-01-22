"""
Generalization tracking for the Seymour feedback loop.

Measures how well labels generalize to new, unseen examples.
For example: after labeling one "tree", can the system recognize other trees?

Key metric: For each label, we track the centroid (average embedding) of all
examples with that label. When a new frame comes in, we measure how close
it is to each known label centroid. If it's close to "tree", the label
generalized successfully.
"""

import time
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


@dataclass
class LabeledSample:
    """A single labeled embedding sample."""
    embedding: np.ndarray
    label: str
    frame_id: int
    timestamp: float


@dataclass 
class LabelCluster:
    """Tracks all embeddings for a single label."""
    label: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Compute the centroid (mean) of all embeddings for this label."""
        if not self.embeddings:
            return None
        return np.mean(np.stack(self.embeddings), axis=0)
    
    @property
    def count(self) -> int:
        return len(self.embeddings)
    
    def add(self, embedding: np.ndarray, frame_id: int, timestamp: float = None):
        """Add a new embedding to this cluster."""
        self.embeddings.append(embedding.copy())
        self.frame_ids.append(frame_id)
        self.timestamps.append(timestamp or time.time())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class GeneralizationTracker:
    """
    Tracks how well labels generalize to new examples.
    
    Usage:
        tracker = GeneralizationTracker()
        
        # When user labels a frame
        tracker.add_label(embedding, "tree", frame_id=42)
        
        # For each new frame, check generalization
        result = tracker.check_generalization(new_embedding)
        print(f"Best match: {result['best_label']} ({result['best_score']:.2f})")
        print(f"Generalized: {result['generalized']}")
        
        # Export all data
        tracker.export_csv("data/labels.csv")
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: Minimum similarity to consider a "match" (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.clusters: Dict[str, LabelCluster] = {}
        
        # History of all generalization checks
        self.history: List[dict] = []
        
        # Running stats
        self._total_checks = 0
        self._successful_generalizations = 0
    
    def add_label(
        self, 
        embedding: np.ndarray, 
        label: str, 
        frame_id: int,
        timestamp: float = None
    ):
        """
        Add a labeled embedding to the tracker.
        
        Args:
            embedding: The CLIP/projected embedding
            label: User-provided label (e.g., "tree")
            frame_id: Frame number
            timestamp: Optional timestamp
        """
        if label not in self.clusters:
            self.clusters[label] = LabelCluster(label=label)
        
        self.clusters[label].add(embedding, frame_id, timestamp)
    
    def check_generalization(
        self, 
        embedding: np.ndarray,
        frame_id: int = -1,
        expected_label: str = None
    ) -> dict:
        """
        Check how well the embedding matches known labels.
        
        Args:
            embedding: New embedding to classify
            frame_id: Frame number (for logging)
            expected_label: If provided, checks if this specific label matches
            
        Returns:
            Dict with:
                - best_label: Most similar label
                - best_score: Similarity score (0-1)
                - all_scores: Dict of {label: score} for all labels
                - generalized: True if best_score >= threshold
                - correct: True if best_label == expected_label (if provided)
        """
        if not self.clusters:
            return {
                'best_label': None,
                'best_score': 0.0,
                'all_scores': {},
                'generalized': False,
                'correct': None,
                'frame_id': frame_id
            }
        
        # Compute similarity to each label's centroid
        scores = {}
        for label, cluster in self.clusters.items():
            centroid = cluster.centroid
            if centroid is not None:
                scores[label] = cosine_similarity(embedding, centroid)
            else:
                scores[label] = 0.0
        
        # Find best match
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        
        # Check if it generalizes (above threshold)
        generalized = best_score >= self.similarity_threshold
        
        # Check if correct (if expected label provided)
        correct = None
        if expected_label is not None:
            correct = (best_label == expected_label)
        
        # Update stats
        self._total_checks += 1
        if generalized:
            self._successful_generalizations += 1
        
        # Record in history
        result = {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'best_label': best_label,
            'best_score': best_score,
            'all_scores': scores.copy(),
            'generalized': generalized,
            'correct': correct,
            'expected_label': expected_label
        }
        self.history.append(result)
        
        return result
    
    def get_generalization_rate(self) -> float:
        """Get the overall generalization success rate."""
        if self._total_checks == 0:
            return 0.0
        return self._successful_generalizations / self._total_checks
    
    def get_label_stats(self) -> Dict[str, dict]:
        """Get statistics for each label."""
        stats = {}
        for label, cluster in self.clusters.items():
            stats[label] = {
                'count': cluster.count,
                'first_frame': cluster.frame_ids[0] if cluster.frame_ids else None,
                'last_frame': cluster.frame_ids[-1] if cluster.frame_ids else None,
            }
        return stats
    
    def get_recent_scores(self, n: int = 50) -> List[float]:
        """Get the most recent generalization scores for plotting."""
        recent = self.history[-n:] if self.history else []
        return [h['best_score'] for h in recent]
    
    def get_recent_history(self, n: int = 10) -> List[dict]:
        """Get the most recent generalization checks."""
        return self.history[-n:] if self.history else []
    
    def export_csv(self, filepath: str):
        """
        Export all labeled data and history to CSV files.
        
        Creates two files:
            - {filepath}_labels.csv: All labeled embeddings
            - {filepath}_history.csv: All generalization checks
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export labels
        labels_file = path.parent / f"{path.stem}_labels.csv"
        with open(labels_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'label', 'timestamp', 'embedding_sample'])
            
            for label, cluster in self.clusters.items():
                for i, (frame_id, ts) in enumerate(zip(cluster.frame_ids, cluster.timestamps)):
                    # Store first 10 embedding values as sample (full embedding would be huge)
                    emb_sample = ','.join(f'{v:.4f}' for v in cluster.embeddings[i][:10])
                    writer.writerow([frame_id, label, ts, emb_sample])
        
        print(f"[generalization] Exported labels to {labels_file}")
        
        # Export history
        history_file = path.parent / f"{path.stem}_history.csv"
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id', 'timestamp', 'best_label', 'best_score', 
                'generalized', 'expected_label', 'correct'
            ])
            
            for h in self.history:
                writer.writerow([
                    h['frame_id'],
                    h['timestamp'],
                    h['best_label'],
                    f"{h['best_score']:.4f}",
                    h['generalized'],
                    h.get('expected_label', ''),
                    h.get('correct', '')
                ])
        
        print(f"[generalization] Exported history to {history_file}")
        
        return str(labels_file), str(history_file)
    
    def export_embeddings_npz(self, filepath: str):
        """Export full embeddings to numpy format for analysis."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for label, cluster in self.clusters.items():
            safe_label = label.replace(' ', '_').replace('/', '_')
            data[f'{safe_label}_embeddings'] = np.stack(cluster.embeddings) if cluster.embeddings else np.array([])
            data[f'{safe_label}_frame_ids'] = np.array(cluster.frame_ids)
            data[f'{safe_label}_timestamps'] = np.array(cluster.timestamps)
        
        np.savez(path, **data)
        print(f"[generalization] Exported embeddings to {path}")
        return str(path)
    
    def summary(self) -> str:
        """Get a text summary of the tracker state."""
        lines = [
            f"Labels: {len(self.clusters)}",
            f"Total samples: {sum(c.count for c in self.clusters.values())}",
            f"Generalization checks: {self._total_checks}",
            f"Success rate: {self.get_generalization_rate():.1%}",
            "",
            "Labels:"
        ]
        
        for label, cluster in self.clusters.items():
            lines.append(f"  '{label}': {cluster.count} samples")
        
        return "\n".join(lines)


# Convenience function for quick testing
def test_generalization():
    """Quick test of the generalization tracker."""
    tracker = GeneralizationTracker(similarity_threshold=0.8)
    
    # Simulate some embeddings
    np.random.seed(42)
    
    # Create "tree" embeddings (similar to each other)
    tree_base = np.random.randn(512)
    for i in range(3):
        emb = tree_base + np.random.randn(512) * 0.1  # Small noise
        tracker.add_label(emb, "tree", frame_id=i)
    
    # Create "rock" embeddings (different from trees)
    rock_base = np.random.randn(512)
    for i in range(3):
        emb = rock_base + np.random.randn(512) * 0.1
        tracker.add_label(emb, "rock", frame_id=i+10)
    
    # Test generalization with a new "tree"
    new_tree = tree_base + np.random.randn(512) * 0.15
    result = tracker.check_generalization(new_tree, frame_id=100, expected_label="tree")
    
    print("Test Result:")
    print(f"  Best label: {result['best_label']}")
    print(f"  Score: {result['best_score']:.3f}")
    print(f"  Generalized: {result['generalized']}")
    print(f"  Correct: {result['correct']}")
    print()
    print(tracker.summary())


if __name__ == "__main__":
    test_generalization()
