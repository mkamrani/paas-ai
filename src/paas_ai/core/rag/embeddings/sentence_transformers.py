"""
SentenceTransformers embedding strategy.
"""

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings

from .base import EmbeddingStrategy
from ..config import EmbeddingConfig


class SentenceTransformersEmbeddingStrategy(EmbeddingStrategy):
    """Strategy for SentenceTransformers embeddings."""
    
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create SentenceTransformers embeddings."""
        params = config.params.copy()
        return SentenceTransformerEmbeddings(
            model_name=config.model_name,
            **params
        )
    
    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate SentenceTransformers embedding configuration."""
        # Basic validation for model name
        if not config.model_name:
            raise ValueError("model_name is required for SentenceTransformers embeddings")
        
        # Check if model name looks valid
        valid_prefixes = ['all-', 'sentence-transformers/', 'paraphrase-', 'distilbert-', 'bert-']
        if not any(config.model_name.startswith(prefix) for prefix in valid_prefixes):
            # Still allow it, just warn
            pass 