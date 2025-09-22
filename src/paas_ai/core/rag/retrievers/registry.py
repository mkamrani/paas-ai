"""
Strategy registry for retrievers.
"""

from typing import Dict, Type
from .base import RetrieverStrategy
from ..config import RetrieverType

# Import all strategies
from .similarity import SimilarityRetrieverStrategy
from .mmr import MMRRetrieverStrategy
from .similarity_score_threshold import SimilarityScoreThresholdRetrieverStrategy
from .ensemble import EnsembleRetrieverStrategy
from .multi_query import MultiQueryRetrieverStrategy
from .parent_document import ParentDocumentRetrieverStrategy


class RetrieverRegistry:
    """Registry for retriever strategies."""
    
    _strategies: Dict[RetrieverType, Type[RetrieverStrategy]] = {
        RetrieverType.SIMILARITY: SimilarityRetrieverStrategy,
        RetrieverType.MMR: MMRRetrieverStrategy,
        RetrieverType.SIMILARITY_SCORE_THRESHOLD: SimilarityScoreThresholdRetrieverStrategy,
        RetrieverType.ENSEMBLE: EnsembleRetrieverStrategy,
        RetrieverType.MULTI_QUERY: MultiQueryRetrieverStrategy,
        RetrieverType.PARENT_DOCUMENT: ParentDocumentRetrieverStrategy,
    }
    
    @classmethod
    def get_strategy(cls, retriever_type: RetrieverType) -> RetrieverStrategy:
        """Get strategy instance for the given retriever type."""
        if retriever_type not in cls._strategies:
            raise ValueError(f"No strategy registered for retriever type: {retriever_type}")
        
        strategy_class = cls._strategies[retriever_type]
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, retriever_type: RetrieverType, strategy_class: Type[RetrieverStrategy]) -> None:
        """Register a new strategy for a retriever type."""
        cls._strategies[retriever_type] = strategy_class
    
    @classmethod
    def list_supported_types(cls) -> list[RetrieverType]:
        """List all supported retriever types."""
        return list(cls._strategies.keys())
