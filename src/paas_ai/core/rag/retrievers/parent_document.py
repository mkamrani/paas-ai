"""
Parent document retriever strategy.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ParentDocumentRetriever

from .base import RetrieverStrategy
from ..config import RetrieverConfig


class ParentDocumentRetrieverStrategy(RetrieverStrategy):
    """Strategy for parent document-based retrieval."""
    
    def create_retriever(
        self, 
        config: RetrieverConfig, 
        vectorstore: VectorStore,
        llm=None
    ) -> BaseRetriever:
        """Create a parent document retriever."""
        params = config.params.copy()
        
        if 'child_splitter' not in params:
            raise ValueError("child_splitter is required for ParentDocumentRetriever")
        
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=params.get('docstore'),
            child_splitter=params['child_splitter'],
            **{k: v for k, v in params.items() if k not in ['docstore', 'child_splitter']}
        )
    
    def validate_config(self, config: RetrieverConfig) -> None:
        """Validate parent document retriever configuration."""
        params = config.params
        
        # Check if params is a dictionary-like object
        if not hasattr(params, 'get') or not hasattr(params, '__contains__'):
            # Try to access params as dict to trigger AttributeError
            _ = params['child_splitter']
        
        if 'child_splitter' not in params:
            raise ValueError("params['child_splitter'] is required for ParentDocumentRetriever")
        
        child_splitter = params['child_splitter']
        if not hasattr(child_splitter, 'split_documents'):
            raise ValueError("params['child_splitter'] must have a split_documents method")
        
        # Validate optional docstore
        if 'docstore' in params:
            docstore = params['docstore']
            if not hasattr(docstore, 'add') or not hasattr(docstore, 'search'):
                raise ValueError("params['docstore'] must have add and search methods")
        
        # Validate optional parent_splitter
        if 'parent_splitter' in params:
            parent_splitter = params['parent_splitter']
            if not hasattr(parent_splitter, 'split_documents'):
                raise ValueError("params['parent_splitter'] must have a split_documents method")
