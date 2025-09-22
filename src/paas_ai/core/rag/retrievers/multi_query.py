"""
Multi-query retriever strategy.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MultiQueryRetriever

from .base import RetrieverStrategy
from ..config import RetrieverConfig


class MultiQueryRetrieverStrategy(RetrieverStrategy):
    """Strategy for multi-query-based retrieval."""
    
    def create_retriever(
        self, 
        config: RetrieverConfig, 
        vectorstore: VectorStore,
        llm=None
    ) -> BaseRetriever:
        """Create a multi-query retriever."""
        if llm is None:
            raise ValueError("LLM is required for MultiQueryRetriever")
        
        search_kwargs = config.search_kwargs.copy()
        params = config.params.copy()
        
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            **params
        )
    
    def validate_config(self, config: RetrieverConfig) -> None:
        """Validate multi-query retriever configuration."""
        k = config.search_kwargs.get('k', 4)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("search_kwargs['k'] must be a positive integer")
        
        # Validate optional parameters
        if 'query_count' in config.params:
            query_count = config.params['query_count']
            if not isinstance(query_count, int) or query_count <= 0:
                raise ValueError("params['query_count'] must be a positive integer")
        
        if 'prompt' in config.params:
            prompt = config.params['prompt']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("params['prompt'] must be a non-empty string")
