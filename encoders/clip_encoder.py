"""
CLIP encoder for generating image embeddings.

Uses open_clip for loading CLIP models and encoding images.
"""

from typing import Optional
import numpy as np

from . import config


class CLIPEncoder:
    """
    CLIP image encoder using open_clip.
    
    Usage:
        encoder = CLIPEncoder()
        encoder.load()
        
        embedding = encoder.encode(frame)  # frame is BGR numpy array
        # embedding is normalized float32 array of shape (512,) for ViT-B-32
    """
    
    def __init__(
        self,
        model_name: str = None,
        pretrained: str = None,
        device: str = None
    ):
        self.model_name = model_name or config.MODEL_NAME
        self.pretrained = pretrained or config.PRETRAINED
        self.device = device or config.get_device()
        
        self._model = None
        self._preprocess = None
        self._loaded = False
        self._embedding_dim = None
    
    def load(self):
        """Load the CLIP model."""
        if self._loaded:
            print("[clip] Model already loaded")
            return
        
        print(f"[clip] Loading {self.model_name} ({self.pretrained}) on {self.device}")
        
        try:
            import torch
            import open_clip
            from PIL import Image
            
            # Store PIL for later use
            self._pil_image = Image
            
            # Load model
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            
            self._model.eval()
            
            # Get embedding dimension by running a dummy input
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(self.device)
                dummy_out = self._model.encode_image(dummy)
                self._embedding_dim = dummy_out.shape[-1]
            
            self._loaded = True
            print(f"[clip] Loaded successfully (dim={self._embedding_dim})")
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import required packages: {e}\n\n"
                "Make sure you have installed:\n"
                "  pip install torch open-clip-torch pillow\n\n"
                "For CUDA support, install PyTorch with CUDA first:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def encode(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode a single frame to CLIP embedding.
        
        Args:
            frame: BGR numpy array (H, W, 3) from OpenCV
            
        Returns:
            Normalized embedding as float32 numpy array (D,)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            import torch
            
            # Convert BGR to RGB
            rgb = frame[:, :, ::-1]
            
            # Convert to PIL Image
            pil_image = self._pil_image.fromarray(rgb)
            
            # Apply CLIP preprocessing
            image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Encode
            with torch.no_grad():
                embedding = self._model.encode_image(image_tensor)
                
                # Normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = embedding.cpu().numpy().squeeze()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"[clip] Encode error: {e}")
            return None
    
    def encode_batch(self, frames: list) -> Optional[np.ndarray]:
        """
        Encode multiple frames in a batch.
        
        Args:
            frames: List of BGR numpy arrays
            
        Returns:
            Normalized embeddings as float32 numpy array (N, D)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not frames:
            return None
        
        try:
            import torch
            
            # Preprocess all frames
            tensors = []
            for frame in frames:
                rgb = frame[:, :, ::-1]
                pil_image = self._pil_image.fromarray(rgb)
                tensor = self._preprocess(pil_image)
                tensors.append(tensor)
            
            # Stack into batch
            batch = torch.stack(tensors).to(self.device)
            
            # Encode
            with torch.no_grad():
                embeddings = self._model.encode_image(batch)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            print(f"[clip] Batch encode error: {e}")
            return None
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim or 512  # Default for ViT-B-32
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def get_info(self) -> dict:
        """Get encoder information."""
        return {
            "model": self.model_name,
            "pretrained": self.pretrained,
            "device": self.device,
            "loaded": self._loaded,
            "embedding_dim": self._embedding_dim
        }

