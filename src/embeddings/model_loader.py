"""Embedding model loader with real and mock implementations."""

import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_settings

logger = logging.getLogger(__name__)

# Global lock for thread-safe model loading
_model_load_lock = threading.Lock()


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        pass


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic hash-based mock embedding model for fast unit testing."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """Generate deterministic pseudo-random embeddings from text hashes."""
        embeddings = []
        for text in texts:
            # Use SHA256 hash of text as seed for deterministic random vector
            hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
            # Convert first 8 bytes to seed (modulo 2**32 for numpy RandomState)
            seed = int.from_bytes(hash_bytes[:8], 'big') % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dimension).astype(np.float32)
            # Normalize to unit length
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Wrapper around sentence-transformers SentenceTransformer."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.cache_dir = cache_dir
        self.normalize_embeddings = normalize_embeddings
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = threading.Lock()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=self.cache_dir,
                    )
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


def load_embedding_model(
    model_name: Optional[str] = None,
    use_mock: bool = False,
    settings=None,
) -> EmbeddingModel:
    """Factory function to load embedding model."""
    if settings is None:
        settings = get_settings()

    if use_mock:
        logger.info("Using MockEmbeddingModel for testing")
        return MockEmbeddingModel(dimension=settings.embedding.model_dim)

    name = model_name or settings.embedding.model_name

    # Check for mock model name explicitly
    if name == "mock":
        logger.info("Using MockEmbeddingModel for testing (model_name=mock)")
        return MockEmbeddingModel(dimension=settings.embedding.model_dim)

    logger.info(f"Loading real embedding model: {name}")

    try:
        return SentenceTransformerEmbeddingModel(
            model_name=name,
            device=settings.embedding.device,
            cache_dir=settings.embedding.cache_dir,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )
    except Exception as e:
        logger.warning(f"Failed to load {name}: {e}. Falling back to {settings.embedding.alternative_models.get('bge_small')}")
        fallback = settings.embedding.alternative_models.get('bge_small', 'BAAI/bge-small-en-v1.5')
        return SentenceTransformerEmbeddingModel(
            model_name=fallback,
            device=settings.embedding.device,
            cache_dir=settings.embedding.cache_dir,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )