"""
Chroma vector store strategy.
"""

from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_chroma import Chroma

from .base import VectorStoreStrategy
from ...config.schemas import VectorStoreConfig


class ChromaVectorStoreStrategy(VectorStoreStrategy):
    """Strategy for Chroma vector stores."""
    
    def create_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None
    ) -> VectorStore:
        """Create a Chroma vector store."""
        persist_directory = None
        if config.persist_directory:
            persist_directory = str(config.persist_directory)
            # Ensure directory exists
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        params = config.params.copy()
        
        if documents:
            return Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=config.collection_name,
                persist_directory=persist_directory,
                **params
            )
        else:
            return Chroma(
                embedding_function=embeddings,
                collection_name=config.collection_name,
                persist_directory=persist_directory,
                **params
            )
    
    def load_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings
    ) -> Optional[VectorStore]:
        """Load an existing Chroma vector store."""
        if not config.persist_directory:
            return None
        
        # Handle both string and Path types
        persist_dir = Path(config.persist_directory)
        if not persist_dir.exists():
            return None
        
        try:
            params = config.params.copy()
            return Chroma(
                embedding_function=embeddings,
                collection_name=config.collection_name,
                persist_directory=str(persist_dir),
                **params
            )
        except Exception:
            return None
    
    def validate_config(self, config: VectorStoreConfig) -> None:
        """Validate Chroma vector store configuration."""
        if not config.collection_name:
            raise ValueError("collection_name is required for Chroma")
        
        # Validate collection name format
        # Remove hyphens and underscores, then check if remaining characters are alphanumeric
        cleaned_name = config.collection_name.replace('_', '').replace('-', '')
        if not cleaned_name.isalnum():
            raise ValueError("collection_name must contain only alphanumeric characters, hyphens, and underscores") 