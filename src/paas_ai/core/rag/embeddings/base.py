"""
Base strategy interface for embeddings.
"""

from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

from ..config import EmbeddingConfig


class EmbeddingStrategy(ABC):
    """Base strategy interface for embeddings."""
    
    @abstractmethod
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create embeddings based on configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate configuration for this embedding strategy."""
        pass 