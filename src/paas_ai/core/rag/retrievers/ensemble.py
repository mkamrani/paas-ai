"""
Ensemble retriever strategy.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever

from .base import RetrieverStrategy
from ..config import RetrieverConfig


class EnsembleRetrieverStrategy(RetrieverStrategy):
    """Strategy for ensemble-based retrieval."""
    
    def create_retriever(
        self, 
        config: RetrieverConfig, 
        vectorstore: VectorStore,
        llm=None
    ) -> BaseRetriever:
        """Create an ensemble retriever."""
        search_kwargs = config.search_kwargs.copy()
        params = config.params.copy()
        
        # Create multiple retrievers for ensemble
        similarity_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        
        weights = params.get('weights', [0.5, 0.5])
        return EnsembleRetriever(
            retrievers=[similarity_retriever, mmr_retriever],
            weights=weights,
            **{k: v for k, v in params.items() if k != 'weights'}
        )
    
    def validate_config(self, config: RetrieverConfig) -> None:
        """Validate ensemble retriever configuration."""
        k = config.search_kwargs.get('k', 4)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("search_kwargs['k'] must be a positive integer")
        
        weights = config.params.get('weights', [0.5, 0.5])
        if not isinstance(weights, list) or len(weights) != 2:
            raise ValueError("params['weights'] must be a list of 2 numbers")
        
        for weight in weights:
            if not isinstance(weight, (int, float)) or not (0 <= weight <= 1):
                raise ValueError("All weights must be numbers between 0 and 1")
        
        if abs(sum(weights) - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError("Weights must sum to 1.0")
