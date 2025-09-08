"""
Factory for creating retrievers based on configuration.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
)

from ..config import RetrieverConfig, RetrieverType


class RetrieverFactory:
    """Factory for creating retrievers based on configuration."""
    
    @staticmethod
    def create_retriever(
        config: RetrieverConfig,
        vectorstore: VectorStore,
        llm = None
    ) -> BaseRetriever:
        """Create a retriever based on configuration."""
        retriever_type = config.type
        search_kwargs = config.search_kwargs.copy()
        params = config.params.copy()
        
        if retriever_type == RetrieverType.SIMILARITY:
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs,
                **params
            )
        
        elif retriever_type == RetrieverType.MMR:
            return vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs,
                **params
            )
        
        elif retriever_type == RetrieverType.SIMILARITY_SCORE_THRESHOLD:
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs,
                **params
            )
        
        elif retriever_type == RetrieverType.ENSEMBLE:
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
        
        elif retriever_type == RetrieverType.MULTI_QUERY:
            if llm is None:
                raise ValueError("LLM is required for MultiQueryRetriever")
            
            base_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            
            return MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
                **params
            )
        
        elif retriever_type == RetrieverType.PARENT_DOCUMENT:
            if 'child_splitter' not in params:
                raise ValueError("child_splitter is required for ParentDocumentRetriever")
            
            return ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=params.get('docstore'),
                child_splitter=params['child_splitter'],
                **{k: v for k, v in params.items() if k not in ['docstore', 'child_splitter']}
            )
        
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}") 