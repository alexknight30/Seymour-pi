"""
LLM interface for visual understanding.

Supports multiple providers:
  - OpenAI (GPT-4, etc.)
  - Anthropic (Claude)
  - Local (Ollama)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import json

import numpy as np

from . import config


class LLMInterface(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def label_embedding(
        self,
        embedding: np.ndarray,
        context: str = ""
    ) -> str:
        """
        Generate a label/description for a visual embedding.
        
        Args:
            embedding: Projected visual embedding
            context: Optional context about what we're looking at
            
        Returns:
            Text label/description
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Get text embedding (for training projection layer).
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        pass


class OpenAILLM(LLMInterface):
    """OpenAI GPT interface."""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.get_llm_model()
        self.api_key = api_key or config.OPENAI_API_KEY
        self._client = None
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        return response.choices[0].message.content
    
    def label_embedding(
        self,
        embedding: np.ndarray,
        context: str = ""
    ) -> str:
        # Convert embedding to a representation the LLM can understand
        # We describe the embedding's key features
        
        embedding_summary = self._summarize_embedding(embedding)
        
        prompt = f"""You are analyzing a visual embedding from a camera feed.

Embedding summary:
{embedding_summary}

{f"Context: {context}" if context else ""}

Based on this visual embedding, provide a brief label (1-5 words) describing what might be in the scene.
Just output the label, nothing else."""

        return self.generate(prompt, max_tokens=20, temperature=0.3)
    
    def _summarize_embedding(self, embedding: np.ndarray) -> str:
        """Create a text summary of embedding statistics."""
        return f"""- Dimension: {len(embedding)}
- Mean: {embedding.mean():.4f}
- Std: {embedding.std():.4f}
- Min: {embedding.min():.4f}
- Max: {embedding.max():.4f}
- Sparsity: {(np.abs(embedding) < 0.1).mean():.2%}
- Top-5 indices: {np.argsort(np.abs(embedding))[-5:][::-1].tolist()}"""

    def embed_text(self, text: str) -> np.ndarray:
        client = self._get_client()
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        return np.array(response.data[0].embedding, dtype=np.float32)


class AnthropicLLM(LLMInterface):
    """Anthropic Claude interface."""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.get_llm_model()
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self._client = None
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
    
    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        
        response = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 256),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def label_embedding(
        self,
        embedding: np.ndarray,
        context: str = ""
    ) -> str:
        embedding_summary = self._summarize_embedding(embedding)
        
        prompt = f"""You are analyzing a visual embedding from a camera feed.

Embedding summary:
{embedding_summary}

{f"Context: {context}" if context else ""}

Based on this visual embedding, provide a brief label (1-5 words) describing what might be in the scene.
Just output the label, nothing else."""

        return self.generate(prompt, max_tokens=20)
    
    def _summarize_embedding(self, embedding: np.ndarray) -> str:
        return f"""- Dimension: {len(embedding)}
- Mean: {embedding.mean():.4f}
- Std: {embedding.std():.4f}
- Min: {embedding.min():.4f}
- Max: {embedding.max():.4f}
- Sparsity: {(np.abs(embedding) < 0.1).mean():.2%}"""

    def embed_text(self, text: str) -> np.ndarray:
        # Anthropic doesn't have a native embedding API
        # We'll use a workaround or raise an error
        raise NotImplementedError(
            "Anthropic doesn't provide text embeddings. "
            "Use OpenAI or local embeddings for training."
        )


class LocalLLM(LLMInterface):
    """Local LLM via Ollama."""
    
    def __init__(self, model: str = None, base_url: str = "http://localhost:11434"):
        self.model = model or config.get_llm_model()
        self.base_url = base_url
    
    def generate(self, prompt: str, **kwargs) -> str:
        import urllib.request
        import urllib.error
        
        url = f"{self.base_url}/api/generate"
        
        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }).encode()
        
        try:
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")
                
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )
    
    def label_embedding(
        self,
        embedding: np.ndarray,
        context: str = ""
    ) -> str:
        embedding_summary = self._summarize_embedding(embedding)
        
        prompt = f"""You are analyzing a visual embedding from a camera feed.

Embedding summary:
{embedding_summary}

{f"Context: {context}" if context else ""}

Based on this visual embedding, provide a brief label (1-5 words) describing what might be in the scene.
Just output the label, nothing else."""

        return self.generate(prompt, max_tokens=20, temperature=0.3)
    
    def _summarize_embedding(self, embedding: np.ndarray) -> str:
        return f"""- Dimension: {len(embedding)}
- Mean: {embedding.mean():.4f}
- Std: {embedding.std():.4f}
- Min: {embedding.min():.4f}
- Max: {embedding.max():.4f}"""

    def embed_text(self, text: str) -> np.ndarray:
        """Get text embedding using Ollama's embedding endpoint."""
        import urllib.request
        
        url = f"{self.base_url}/api/embeddings"
        
        data = json.dumps({
            "model": self.model,
            "prompt": text
        }).encode()
        
        try:
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode())
                return np.array(result.get("embedding", []), dtype=np.float32)
                
        except Exception as e:
            print(f"[llm] Embedding error: {e}")
            # Return zero vector as fallback
            return np.zeros(config.PROJECTION_DIM, dtype=np.float32)


def create_llm(provider: str = None, **kwargs) -> LLMInterface:
    """
    Factory function to create LLM instance.
    
    Args:
        provider: 'openai', 'anthropic', or 'local'
        **kwargs: Provider-specific arguments
        
    Returns:
        LLMInterface instance
    """
    provider = provider or config.LLM_PROVIDER
    
    if provider == "openai":
        return OpenAILLM(**kwargs)
    elif provider == "anthropic":
        return AnthropicLLM(**kwargs)
    elif provider == "local":
        return LocalLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

