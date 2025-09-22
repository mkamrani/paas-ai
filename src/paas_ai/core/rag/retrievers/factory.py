"""
Factory for creating retrievers based on configuration.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from ..config import RetrieverConfig
from .registry import RetrieverRegistry


class RetrieverFactory:
    """Factory for creating retrievers based on configuration using registry pattern."""
    
    @staticmethod
    def create_retriever(
        config: RetrieverConfig,
        vectorstore: VectorStore,
        llm=None
    ) -> BaseRetriever:
        """Create a retriever based on configuration using registry pattern."""
        # Get strategy from registry
        strategy = RetrieverRegistry.get_strategy(config.type)
        
        # Validate configuration before creating retriever
        strategy.validate_config(config)
        
        # Create retriever using the appropriate strategy
        return strategy.create_retriever(config, vectorstore, llm)
    
    @staticmethod
    def register_strategy(retriever_type, strategy_class):
        """Register a new strategy for a retriever type."""
        RetrieverRegistry.register_strategy(retriever_type, strategy_class)
    
    @staticmethod
    def list_supported_types():
        """List all supported retriever types."""
        return RetrieverRegistry.list_supported_types() 