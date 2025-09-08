"""
Similarity retriever strategy.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from .base import RetrieverStrategy
from ..config import RetrieverConfig


class SimilarityRetrieverStrategy(RetrieverStrategy):
    """Strategy for similarity-based retrieval."""
    
    def create_retriever(
        self, 
        config: RetrieverConfig, 
        vectorstore: VectorStore,
        llm=None
    ) -> BaseRetriever:
        """Create a similarity retriever."""
        search_kwargs = config.search_kwargs.copy()
        params = config.params.copy()
        
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
            **params
        )
    
    def validate_config(self, config: RetrieverConfig) -> None:
        """Validate similarity retriever configuration."""
        k = config.search_kwargs.get('k', 4)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("search_kwargs['k'] must be a positive integer") 