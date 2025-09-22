"""
Factory for creating vector stores based on configuration.
"""

from typing import List, Optional, Dict, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from paas_ai.core.config.schemas import VectorStoreConfig, VectorStoreType
from .base import VectorStoreStrategy
from .chroma import ChromaVectorStoreStrategy
from .faiss import FAISSVectorStoreStrategy
from .pinecone import PineconeVectorStoreStrategy


class VectorStoreFactory:
    """Factory for creating vector stores based on configuration."""
    
    # Registry of available strategies
    _strategies: Dict[VectorStoreType, Type[VectorStoreStrategy]] = {
        VectorStoreType.CHROMA: ChromaVectorStoreStrategy,
        VectorStoreType.FAISS: FAISSVectorStoreStrategy,
        VectorStoreType.PINECONE: PineconeVectorStoreStrategy,
    }
    
    @classmethod
    def create_vectorstore(
        cls,
        config: VectorStoreConfig,
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None
    ) -> VectorStore:
        """Create a vector store based on configuration."""
        # Validate configuration first
        cls._validate_config(config)
        
        # Get the appropriate strategy
        strategy_class = cls._strategies.get(config.type)
        if strategy_class is None:
            raise ValueError(f"Unsupported vector store type: {config.type}")
        
        # Create strategy instance and delegate to it
        strategy = strategy_class()
        return strategy.create_vectorstore(config, embeddings, documents)
    
    @classmethod
    def load_vectorstore(
        cls,
        config: VectorStoreConfig,
        embeddings: Embeddings
    ) -> Optional[VectorStore]:
        """Load an existing vector store from disk."""
        # Get the appropriate strategy
        strategy_class = cls._strategies.get(config.type)
        if strategy_class is None:
            return None
        
        # Create strategy instance and delegate to it
        strategy = strategy_class()
        return strategy.load_vectorstore(config, embeddings)
    
    @classmethod
    def _validate_config(cls, config: VectorStoreConfig) -> None:
        """Validate configuration using the appropriate strategy."""
        strategy_class = cls._strategies.get(config.type)
        if strategy_class is None:
            raise ValueError(f"Unsupported vector store type: {config.type}")
        
        # Create strategy instance and validate
        strategy = strategy_class()
        strategy.validate_config(config)
    
    @classmethod
    def register_strategy(
        cls, 
        vectorstore_type: VectorStoreType, 
        strategy_class: Type[VectorStoreStrategy]
    ) -> None:
        """Register a new vector store strategy."""
        if vectorstore_type is None:
            raise TypeError("vectorstore_type cannot be None")
        if strategy_class is None:
            raise TypeError("strategy_class cannot be None")
        if not isinstance(strategy_class, type):
            raise TypeError("strategy_class must be a class")
        
        cls._strategies[vectorstore_type] = strategy_class
    
    @classmethod
    def get_available_types(cls) -> List[VectorStoreType]:
        """Get list of available vector store types."""
        if cls._strategies is None:
            raise TypeError("_strategies is None")
        return list(cls._strategies.keys()) 