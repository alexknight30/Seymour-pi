"""
Feedback loop for encoder refinement.

This module handles the cycle:
  1. Receive visual embedding from encoder
  2. Get label from LLM
  3. Convert label to target embedding
  4. Update projection layer to align visual -> text
  5. (Optional) Fine-tune encoder itself
"""

import time
from typing import Optional, Tuple, List, Callable
from pathlib import Path

import numpy as np

from . import config
from .projection import ProjectionLayer, FeedbackBuffer
from .llm import LLMInterface, create_llm


class FeedbackLoop:
    """
    Manages the encoder <-> LLM feedback loop.
    
    The loop works as follows:
    
    1. Visual embedding comes in from CLIP encoder
    2. Projection layer maps it to LLM-compatible space
    3. LLM generates a label/description
    4. Label is converted to text embedding
    5. Projection layer is updated to better align visual -> text
    6. Over time, the projection learns to produce better representations
    
    Optionally, accumulated feedback can be used to fine-tune the
    CLIP encoder itself (requires more compute).
    
    Usage:
        loop = FeedbackLoop()
        loop.start()
        
        # For each frame
        label = loop.process(clip_embedding)
        print(f"LLM says: {label}")
        
        # Periodically train
        if loop.should_train():
            loop.train_step()
        
        loop.save()
    """
    
    def __init__(
        self,
        llm: LLMInterface = None,
        projection: ProjectionLayer = None,
        auto_train: bool = True,
        train_interval: int = 50,  # Train every N samples
    ):
        self.llm = llm
        self.projection = projection or ProjectionLayer()
        self.auto_train = auto_train
        self.train_interval = train_interval
        
        self.feedback_buffer = FeedbackBuffer()
        
        self._sample_count = 0
        self._train_count = 0
        self._last_train_time = 0
        self._running = False
        
        # Callbacks
        self._on_label: Optional[Callable[[str, np.ndarray], None]] = None
        self._on_train: Optional[Callable[[float], None]] = None
    
    def start(self):
        """Initialize the feedback loop."""
        print("[feedback] Starting feedback loop...")
        
        # Initialize LLM if not provided
        if self.llm is None:
            try:
                self.llm = create_llm()
                print(f"[feedback] Using LLM provider: {config.LLM_PROVIDER}")
            except Exception as e:
                print(f"[feedback] Warning: Could not initialize LLM: {e}")
                print("[feedback] Running in embedding-only mode (no labels)")
        
        # Load projection weights if available
        self.projection.load()
        
        # Load feedback buffer if available
        buffer_path = Path("data/feedback_buffer.npz")
        if buffer_path.exists():
            self.feedback_buffer.load(str(buffer_path))
        
        self._running = True
        print("[feedback] Ready")
    
    def stop(self):
        """Stop and save state."""
        print("[feedback] Stopping...")
        
        self.projection.save()
        
        # Save feedback buffer
        Path("data").mkdir(exist_ok=True)
        self.feedback_buffer.save("data/feedback_buffer.npz")
        
        self._running = False
        print("[feedback] Stopped")
    
    def process(
        self,
        embedding: np.ndarray,
        context: str = "",
        get_label: bool = True
    ) -> Tuple[str, np.ndarray]:
        """
        Process a visual embedding through the feedback loop.
        
        Args:
            embedding: CLIP embedding from encoder
            context: Optional context for LLM
            get_label: Whether to query LLM for label
            
        Returns:
            (label, projected_embedding)
        """
        self._sample_count += 1
        
        # Project embedding
        projected = self.projection.project(embedding)
        
        # Get label from LLM
        label = ""
        if get_label and self.llm is not None:
            try:
                label = self.llm.label_embedding(projected, context)
                label = label.strip()
                
                # Store feedback pair
                self.feedback_buffer.add(embedding, label)
                
                # Callback
                if self._on_label:
                    self._on_label(label, embedding)
                    
            except Exception as e:
                print(f"[feedback] LLM error: {e}")
        
        # Auto-train if enabled
        if self.auto_train and self.should_train():
            loss = self.train_step()
            if self._on_train:
                self._on_train(loss)
        
        return label, projected
    
    def should_train(self) -> bool:
        """Check if we should run a training step."""
        return (
            self._sample_count > 0 and
            self._sample_count % self.train_interval == 0 and
            len(self.feedback_buffer) >= config.FEEDBACK_BATCH_SIZE
        )
    
    def train_step(self) -> float:
        """
        Run one training step on accumulated feedback.
        
        Returns:
            Loss value
        """
        if len(self.feedback_buffer) < 2:
            return 0.0
        
        # Get batch
        embeddings, labels = self.feedback_buffer.get_batch()
        
        if len(embeddings) == 0:
            return 0.0
        
        # Convert labels to target embeddings
        target_embeddings = []
        for label in labels:
            try:
                if self.llm is not None:
                    target = self.llm.embed_text(label)
                else:
                    # Fallback: use random target (won't train well)
                    target = np.random.randn(config.PROJECTION_DIM).astype(np.float32)
                target_embeddings.append(target)
            except Exception as e:
                print(f"[feedback] Could not embed label '{label}': {e}")
                # Use zero vector as fallback
                target_embeddings.append(np.zeros(config.PROJECTION_DIM, dtype=np.float32))
        
        target_embeddings = np.stack(target_embeddings)
        
        # Update projection
        loss = self.projection.update(embeddings, target_embeddings, labels)
        
        self._train_count += 1
        self._last_train_time = time.time()
        
        print(f"[feedback] Train step {self._train_count}: loss={loss:.4f}")
        
        return loss
    
    def add_manual_label(self, embedding: np.ndarray, label: str):
        """
        Manually add a labeled embedding (for supervised feedback).
        
        Args:
            embedding: CLIP embedding
            label: Human-provided label
        """
        self.feedback_buffer.add(embedding, label)
        print(f"[feedback] Added manual label: '{label}'")
    
    def set_on_label(self, callback: Callable[[str, np.ndarray], None]):
        """Set callback for when a label is generated."""
        self._on_label = callback
    
    def set_on_train(self, callback: Callable[[float], None]):
        """Set callback for when training occurs."""
        self._on_train = callback
    
    def get_stats(self) -> dict:
        """Get feedback loop statistics."""
        return {
            "samples_processed": self._sample_count,
            "train_steps": self._train_count,
            "buffer_size": len(self.feedback_buffer),
            "running": self._running
        }


class EncoderFineTuner:
    """
    Fine-tunes the CLIP encoder based on accumulated feedback.
    
    This is more heavyweight than just training the projection layer,
    but can improve the encoder's representations directly.
    
    WARNING: This modifies the encoder weights. Save a backup first!
    """
    
    def __init__(
        self,
        encoder,  # CLIPEncoder instance
        learning_rate: float = 1e-5,
        freeze_layers: int = 10  # Freeze first N layers
    ):
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.freeze_layers = freeze_layers
        
        self._optimizer = None
        self._setup_done = False
    
    def setup(self):
        """Prepare encoder for fine-tuning."""
        if self._setup_done:
            return
        
        import torch
        
        # Access the underlying model
        model = self.encoder._model
        
        # Freeze early layers
        frozen = 0
        for name, param in model.named_parameters():
            if frozen < self.freeze_layers:
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True
        
        # Create optimizer for trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate
        )
        
        self._setup_done = True
        print(f"[fine-tuner] Setup complete. {len(trainable_params)} trainable parameters.")
    
    def train_step(
        self,
        images: List[np.ndarray],
        labels: List[str]
    ) -> float:
        """
        Fine-tune encoder on image-label pairs.
        
        Uses contrastive loss similar to CLIP training.
        
        Args:
            images: List of BGR numpy arrays
            labels: Corresponding text labels
            
        Returns:
            Loss value
        """
        if not self._setup_done:
            self.setup()
        
        import torch
        import torch.nn.functional as F
        
        # This is a simplified version of CLIP's contrastive training
        # Full implementation would require more infrastructure
        
        # For now, raise NotImplemented as this requires significant work
        raise NotImplementedError(
            "Full encoder fine-tuning is not yet implemented. "
            "Use the projection layer training instead."
        )
    
    def save(self, path: str):
        """Save fine-tuned encoder weights."""
        import torch
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            self.encoder._model.state_dict(),
            path
        )
        print(f"[fine-tuner] Saved encoder to {path}")

