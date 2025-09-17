"""
SentenceTransformers embedding strategy.
"""

try:
    # Try to use the new langchain-huggingface package
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to the deprecated langchain-community version
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import warnings
    warnings.warn(
        "Using deprecated HuggingFaceEmbeddings from langchain-community. "
        "Consider installing langchain-huggingface: pip install langchain-huggingface",
        DeprecationWarning,
        stacklevel=2
    )

from langchain_core.embeddings import Embeddings

from .base import EmbeddingStrategy
from ..config import EmbeddingConfig


class SentenceTransformersEmbeddingStrategy(EmbeddingStrategy):
    """Strategy for SentenceTransformers embeddings."""
    
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create SentenceTransformers embeddings."""
        params = config.params.copy()
        
        # Use HuggingFaceEmbeddings with SentenceTransformers models
        # This avoids the meta tensor issue with SentenceTransformerEmbeddings
        return HuggingFaceEmbeddings(
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
            import warnings
            warnings.warn(
                f"Model name '{config.model_name}' doesn't match common SentenceTransformers patterns. "
                f"Expected prefixes: {valid_prefixes}",
                UserWarning
            ) 