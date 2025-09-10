"""
Factory for creating vector stores based on configuration.
"""

from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Vector stores
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_community.vectorstores import FAISS

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

from ..config import VectorStoreConfig, VectorStoreType


class VectorStoreFactory:
    """Factory for creating vector stores based on configuration."""
    
    @staticmethod
    def create_vectorstore(
        config: VectorStoreConfig,
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None
    ) -> VectorStore:
        """Create a vector store based on configuration."""
        vectorstore_type = config.type
        params = config.params.copy()
        
        if vectorstore_type == VectorStoreType.CHROMA:
            persist_directory = None
            if config.persist_directory:
                persist_directory = str(config.persist_directory)
                # Ensure directory exists
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
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
        
        elif vectorstore_type == VectorStoreType.FAISS:
            if documents:
                vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    **params
                )
            else:
                # Create empty FAISS index
                import faiss
                import numpy as np
                
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
                    index_to_docstore_id={},
                    **params
                )
            
            # Save if persist directory is specified
            if config.persist_directory:
                save_path = str(config.persist_directory)
                Path(save_path).mkdir(parents=True, exist_ok=True)
                vectorstore.save_local(save_path)
            
            return vectorstore
        
        elif vectorstore_type == VectorStoreType.PINECONE:
            if PineconeVectorStore is None:
                raise ImportError("Pinecone integration requires pinecone-client package")
            
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
        
        else:
            raise ValueError(f"Unsupported vector store type: {vectorstore_type}")
    
    @staticmethod
    def load_vectorstore(
        config: VectorStoreConfig,
        embeddings: Embeddings
    ) -> Optional[VectorStore]:
        """Load an existing vector store from disk."""
        if not config.persist_directory:
            return None
            
        # Handle both string and Path types
        persist_dir = Path(config.persist_directory)
        if not persist_dir.exists():
            return None
        
        vectorstore_type = config.type
        params = config.params.copy()
        
        try:
            if vectorstore_type == VectorStoreType.CHROMA:
                return Chroma(
                    embedding_function=embeddings,
                    collection_name=config.collection_name,
                    persist_directory=str(persist_dir),
                    **params
                )
            
            elif vectorstore_type == VectorStoreType.FAISS:
                return FAISS.load_local(
                    folder_path=str(persist_dir),
                    embeddings=embeddings,
                    **params
                )
            
            else:
                return None
                
        except Exception:
            return None 