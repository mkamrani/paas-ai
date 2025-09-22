"""
MMR (Maximum Marginal Relevance) retriever strategy.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from .base import RetrieverStrategy
from ..config import RetrieverConfig


class MMRRetrieverStrategy(RetrieverStrategy):
    """Strategy for MMR-based retrieval."""
    
    def create_retriever(
        self, 
        config: RetrieverConfig, 
        vectorstore: VectorStore,
        llm=None
    ) -> BaseRetriever:
        """Create an MMR retriever."""
        search_kwargs = config.search_kwargs.copy()
        params = config.params.copy()
        
        # Add default values if not provided
        if 'fetch_k' not in search_kwargs:
            search_kwargs['fetch_k'] = 20
        if 'lambda_mult' not in search_kwargs:
            search_kwargs['lambda_mult'] = 0.5
        
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
            **params
        )
    
    def validate_config(self, config: RetrieverConfig) -> None:
        """Validate MMR retriever configuration."""
        k = config.search_kwargs.get('k', 4)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("search_kwargs['k'] must be a positive integer")
        
        fetch_k = config.search_kwargs.get('fetch_k', 20)
        if not isinstance(fetch_k, int) or fetch_k <= 0:
            raise ValueError("search_kwargs['fetch_k'] must be a positive integer")
        
        if fetch_k < k:
            raise ValueError("search_kwargs['fetch_k'] must be >= search_kwargs['k']")
        
        lambda_mult = config.search_kwargs.get('lambda_mult', 0.5)
        if not isinstance(lambda_mult, (int, float)) or not (0 <= lambda_mult <= 1):
            raise ValueError("search_kwargs['lambda_mult'] must be a number between 0 and 1")
