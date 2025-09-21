"""
Strategy registry for embeddings.
"""

from typing import Dict, Type
from .base import EmbeddingStrategy
from ..config import EmbeddingType

# Import strategies
from .openai import OpenAIEmbeddingStrategy
from .sentence_transformers import SentenceTransformersEmbeddingStrategy


class EmbeddingRegistry:
    """Registry for embedding strategies."""
    
    _strategies: Dict[EmbeddingType, Type[EmbeddingStrategy]] = {
        EmbeddingType.OPENAI: OpenAIEmbeddingStrategy,
        EmbeddingType.SENTENCE_TRANSFORMERS: SentenceTransformersEmbeddingStrategy,
        # TODO: Add remaining strategies (HuggingFace, Cohere, Azure OpenAI)
    }
    
    @classmethod
    def get_strategy(cls, embedding_type: EmbeddingType) -> EmbeddingStrategy:
        """Get strategy instance for the given embedding type."""
        if embedding_type not in cls._strategies:
            raise ValueError(f"No strategy registered for embedding type: {embedding_type}")
        
        strategy_class = cls._strategies[embedding_type]
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, embedding_type: EmbeddingType, strategy_class: Type[EmbeddingStrategy]) -> None:
        """Register a new strategy for an embedding type."""
        cls._strategies[embedding_type] = strategy_class
    
    @classmethod
    def list_supported_types(cls) -> list[EmbeddingType]:
        """List all supported embedding types."""
        return list(cls._strategies.keys())
    
    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset registry to default strategies (for testing)."""
        cls._strategies = {
            EmbeddingType.OPENAI: OpenAIEmbeddingStrategy,
            EmbeddingType.SENTENCE_TRANSFORMERS: SentenceTransformersEmbeddingStrategy,
            # TODO: Add remaining strategies (HuggingFace, Cohere, Azure OpenAI)
        } 