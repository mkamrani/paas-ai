"""
FAISS vector store strategy.
"""

from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

from .base import VectorStoreStrategy
from ...config.schemas import VectorStoreConfig


class FAISSVectorStoreStrategy(VectorStoreStrategy):
    """Strategy for FAISS vector stores."""
    
    def create_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None
    ) -> VectorStore:
        """Create a FAISS vector store."""
        params = config.params.copy()
        
        if documents:
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
                **params
            )
        else:
            # Create empty FAISS index
            # Get embedding dimension
            sample_text = "sample text for dimension calculation"
            sample_embedding = embeddings.embed_query(sample_text)
            dimension = len(sample_embedding)
            
            # Create empty index
            index = faiss.IndexFlatL2(dimension)
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore={},
                index_to_docstore_id={}
            )
        
        # Save if persist directory is specified
        if config.persist_directory:
            save_path = str(config.persist_directory)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(save_path)
        
        return vectorstore
    
    def load_vectorstore(
        self, 
        config: VectorStoreConfig, 
        embeddings: Embeddings
    ) -> Optional[VectorStore]:
        """Load an existing FAISS vector store."""
        if not config.persist_directory:
            return None
        
        # Handle both string and Path types
        persist_dir = Path(config.persist_directory)
        if not persist_dir.exists():
            return None
        
        try:
            params = config.params.copy()
            return FAISS.load_local(
                folder_path=str(persist_dir),
                embeddings=embeddings,
                **params
            )
        except Exception:
            return None
    
    def validate_config(self, config: VectorStoreConfig) -> None:
        """Validate FAISS vector store configuration."""
        # FAISS doesn't require collection_name, but we can validate other params
        persist_directory = getattr(config, 'persist_directory', None)
        if persist_directory:
            persist_path = Path(persist_directory)
            # Check if parent directory is writable
            if persist_path.parent.exists() and not persist_path.parent.is_dir():
                raise ValueError(f"persist_directory parent must be a directory: {persist_path.parent}")
