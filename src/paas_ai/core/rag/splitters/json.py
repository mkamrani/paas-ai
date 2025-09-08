"""
JSON text splitter strategy.
"""

from langchain_text_splitters import RecursiveJsonSplitter

from .base import SplitterStrategy
from ..config import SplitterConfig


class JSONSplitterStrategy(SplitterStrategy):
    """Strategy for JSON-aware text splitting."""
    
    def create_splitter(self, config: SplitterConfig):
        """Create a recursive JSON splitter."""
        params = config.params.copy()
        return RecursiveJsonSplitter(
            max_chunk_size=config.chunk_size,
            **params
        )
    
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate JSON splitter configuration."""
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        # Validate max_depth if provided
        max_depth = config.params.get('max_depth')
        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth < 0:
                raise ValueError("max_depth must be a non-negative integer") 