"""
Factory for creating embeddings using strategy pattern.
"""

from langchain_core.embeddings import Embeddings

from ..config import EmbeddingConfig, EmbeddingType
from .registry import EmbeddingRegistry


class EmbeddingsFactory:
    """Factory for creating embeddings using strategy pattern."""
    
    @staticmethod
    def create_embeddings(config: EmbeddingConfig) -> Embeddings:
        """Create embeddings based on configuration."""
        # Get strategy for this embedding type
        strategy = EmbeddingRegistry.get_strategy(config.type)
        
        # Validate configuration
        strategy.validate_config(config)
        
        # Create and return embeddings
        return strategy.create_embeddings(config)
    
    @staticmethod
    def list_supported_types() -> list[EmbeddingType]:
        """List all supported embedding types."""
        return EmbeddingRegistry.list_supported_types()
    
    @staticmethod 
    def register_strategy(embedding_type: EmbeddingType, strategy_class) -> None:
        """Register a custom strategy for an embedding type."""
        EmbeddingRegistry.register_strategy(embedding_type, strategy_class) 