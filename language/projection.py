"""
Projection layer to convert CLIP embeddings into LLM-compatible representations.

The projection layer learns to map visual embeddings into a space that
the LLM can better understand, and can be fine-tuned based on feedback.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from . import config


class ProjectionLayer:
    """
    Learnable projection from CLIP embedding space to LLM-compatible space.
    
    This is a simple MLP that can be trained/fine-tuned based on
    LLM feedback to improve visual-language alignment.
    
    Architecture:
        CLIP embedding (512) -> Linear -> ReLU -> Linear -> Output (768)
    
    Usage:
        proj = ProjectionLayer()
        proj.load()  # Load pretrained weights if available
        
        llm_input = proj.project(clip_embedding)
        
        # After getting feedback from LLM
        proj.update(embeddings, labels)
        proj.save()
    """
    
    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        hidden_dim: int = None,
        device: str = None
    ):
        self.input_dim = input_dim or config.CLIP_EMBEDDING_DIM
        self.output_dim = output_dim or config.PROJECTION_DIM
        self.hidden_dim = hidden_dim or (self.input_dim + self.output_dim) // 2
        self.device = device or config.get_device()
        
        self._model = None
        self._optimizer = None
        self._loaded = False
    
    def _build_model(self):
        """Build the projection MLP."""
        import torch
        import torch.nn as nn
        
        self._model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        ).to(self.device)
        
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        self._loaded = True
        print(f"[projection] Built model: {self.input_dim} -> {self.hidden_dim} -> {self.output_dim}")
    
    def load(self, path: str = None):
        """Load projection weights from file."""
        if self._model is None:
            self._build_model()
        
        if path is None:
            path = "models/projection.pt"
        
        path = Path(path)
        if path.exists():
            import torch
            checkpoint = torch.load(path, map_location=self.device)
            self._model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                self._optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"[projection] Loaded weights from {path}")
        else:
            print(f"[projection] No weights found at {path}, using random init")
    
    def save(self, path: str = None):
        """Save projection weights to file."""
        if self._model is None:
            print("[projection] No model to save")
            return
        
        if path is None:
            path = "models/projection.pt"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import torch
        torch.save({
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_dim': self.hidden_dim
            }
        }, path)
        print(f"[projection] Saved weights to {path}")
    
    def project(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project CLIP embedding to LLM-compatible space.
        
        Args:
            embedding: CLIP embedding (D,) or batch (N, D)
            
        Returns:
            Projected embedding(s)
        """
        if self._model is None:
            self._build_model()
        
        import torch
        
        # Handle single embedding vs batch
        single = embedding.ndim == 1
        if single:
            embedding = embedding[np.newaxis, :]
        
        # Convert to tensor
        x = torch.from_numpy(embedding).float().to(self.device)
        
        # Project
        with torch.no_grad():
            self._model.eval()
            y = self._model(x)
        
        # Convert back
        result = y.cpu().numpy()
        
        if single:
            result = result.squeeze(0)
        
        return result
    
    def update(
        self,
        embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> float:
        """
        Update projection based on feedback.
        
        This trains the projection to map CLIP embeddings closer to
        target embeddings (which could be text embeddings of labels).
        
        Args:
            embeddings: CLIP embeddings (N, D)
            target_embeddings: Target embeddings to align to (N, D_out)
            labels: Optional text labels (for logging)
            
        Returns:
            Loss value
        """
        if self._model is None:
            self._build_model()
        
        import torch
        import torch.nn.functional as F
        
        # Convert to tensors
        x = torch.from_numpy(embeddings).float().to(self.device)
        y_target = torch.from_numpy(target_embeddings).float().to(self.device)
        
        # Forward pass
        self._model.train()
        y_pred = self._model(x)
        
        # Cosine similarity loss (we want projected embeddings to align with targets)
        # Normalize both
        y_pred_norm = F.normalize(y_pred, dim=-1)
        y_target_norm = F.normalize(y_target, dim=-1)
        
        # Cosine similarity -> we want to maximize, so minimize negative
        similarity = (y_pred_norm * y_target_norm).sum(dim=-1)
        loss = -similarity.mean()  # Negative because we want to maximize similarity
        
        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        return loss.item()
    
    def get_embedding_for_text(self, text: str, clip_model) -> np.ndarray:
        """
        Get text embedding using CLIP's text encoder.
        
        This can be used as a target for training the projection.
        
        Args:
            text: Text string to encode
            clip_model: CLIP model instance (from encoders.clip_encoder)
            
        Returns:
            Text embedding
        """
        # This would use CLIP's text encoder
        # For now, placeholder - will implement with CLIP text encoding
        raise NotImplementedError("Text encoding requires CLIP text encoder integration")


class FeedbackBuffer:
    """
    Buffer for collecting (embedding, label) pairs for batch training.
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._embeddings: List[np.ndarray] = []
        self._labels: List[str] = []
        self._timestamps: List[float] = []
    
    def add(self, embedding: np.ndarray, label: str, timestamp: float = None):
        """Add an (embedding, label) pair."""
        import time
        
        self._embeddings.append(embedding.copy())
        self._labels.append(label)
        self._timestamps.append(timestamp or time.time())
        
        # Remove oldest if over capacity
        while len(self._embeddings) > self.capacity:
            self._embeddings.pop(0)
            self._labels.pop(0)
            self._timestamps.pop(0)
    
    def get_batch(self, batch_size: int = None) -> Tuple[np.ndarray, List[str]]:
        """Get a batch of (embeddings, labels)."""
        batch_size = batch_size or config.FEEDBACK_BATCH_SIZE
        batch_size = min(batch_size, len(self._embeddings))
        
        if batch_size == 0:
            return np.array([]), []
        
        # Get most recent batch
        embeddings = np.stack(self._embeddings[-batch_size:])
        labels = self._labels[-batch_size:]
        
        return embeddings, labels
    
    def clear(self):
        """Clear the buffer."""
        self._embeddings = []
        self._labels = []
        self._timestamps = []
    
    def __len__(self) -> int:
        return len(self._embeddings)
    
    def save(self, path: str):
        """Save buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            path,
            embeddings=np.stack(self._embeddings) if self._embeddings else np.array([]),
            labels=np.array(self._labels),
            timestamps=np.array(self._timestamps)
        )
        print(f"[feedback] Saved {len(self)} pairs to {path}")
    
    def load(self, path: str):
        """Load buffer from disk."""
        path = Path(path)
        if not path.exists():
            print(f"[feedback] No buffer found at {path}")
            return
        
        data = np.load(path, allow_pickle=True)
        
        if len(data['embeddings']) > 0:
            self._embeddings = list(data['embeddings'])
            self._labels = list(data['labels'])
            self._timestamps = list(data['timestamps'])
            print(f"[feedback] Loaded {len(self)} pairs from {path}")

