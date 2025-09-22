"""
Base strategy interface for vector stores.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ...config.schemas import VectorStoreConfig


class VectorStoreStrategy(ABC):
    """Base strategy interface for vector stores."""
    
    @abstractmethod
    def create_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None
    ) -> VectorStore:
        """Create a vector store based on configuration."""
        pass
    
    @abstractmethod
    def load_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings
    ) -> Optional[VectorStore]:
        """Load an existing vector store from disk."""
        pass
    
    @abstractmethod
    def validate_config(self, config: VectorStoreConfig) -> None:
        """Validate configuration for this vector store strategy."""
        pass 