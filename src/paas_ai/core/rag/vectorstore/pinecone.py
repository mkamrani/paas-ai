"""
Pinecone vector store strategy.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

from .base import VectorStoreStrategy
from ...config.schemas import VectorStoreConfig


class PineconeVectorStoreStrategy(VectorStoreStrategy):
    """Strategy for Pinecone vector stores."""
    
    def create_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None
    ) -> VectorStore:
        """Create a Pinecone vector store."""
        if PineconeVectorStore is None:
            raise ImportError("Pinecone integration requires pinecone-client package")
        
        params = config.params.copy()
        
        if documents:
            return PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=config.collection_name,
                **params
            )
        else:
            return PineconeVectorStore(
                embedding=embeddings,
                index_name=config.collection_name,
                **params
            )
    
    def load_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings
    ) -> Optional[VectorStore]:
        """Load an existing Pinecone vector store."""
        if PineconeVectorStore is None:
            return None
        
        try:
            params = config.params.copy()
            return PineconeVectorStore(
                embedding=embeddings,
                index_name=config.collection_name,
                **params
            )
        except Exception:
            return None
    
    def validate_config(self, config: VectorStoreConfig) -> None:
        """Validate Pinecone vector store configuration."""
        if not config.collection_name:
            raise ValueError("collection_name is required for Pinecone")
        
        # Ensure collection_name is a string
        if not isinstance(config.collection_name, str):
            raise TypeError("collection_name must be a string")
        
        # Validate collection name format for Pinecone
        # Pinecone collection names must be lowercase and contain only letters, numbers, and hyphens
        if any(c.isupper() for c in config.collection_name):
            raise ValueError("Pinecone collection_name must be lowercase")
        
        # Check for valid characters (ASCII letters, numbers, hyphens only)
        if not all(c.isascii() and (c.isalnum() or c == '-') for c in config.collection_name):
            raise ValueError("Pinecone collection_name must contain only letters, numbers, and hyphens")
        
        # Check length (Pinecone has a 45 character limit)
        if len(config.collection_name) > 45:
            raise ValueError("Pinecone collection_name must be 45 characters or less")
