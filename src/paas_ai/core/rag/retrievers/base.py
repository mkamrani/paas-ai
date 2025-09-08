"""
Base strategy interface for retrievers.
"""

from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from ..config import RetrieverConfig


class RetrieverStrategy(ABC):
    """Base strategy interface for retrievers."""
    
    @abstractmethod
    def create_retriever(
        self, 
        config: RetrieverConfig, 
        vectorstore: VectorStore,
        llm=None
    ) -> BaseRetriever:
        """Create a retriever based on configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: RetrieverConfig) -> None:
        """Validate configuration for this retriever strategy."""
        pass 