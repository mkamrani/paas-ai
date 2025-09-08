"""
Recursive character text splitter strategy.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import SplitterStrategy
from ..config import SplitterConfig


class RecursiveCharacterSplitterStrategy(SplitterStrategy):
    """Strategy for recursive character-based text splitting."""
    
    def create_splitter(self, config: SplitterConfig):
        """Create a recursive character text splitter."""
        params = config.params.copy()
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=params.get('separators'),
            **{k: v for k, v in params.items() if k != 'separators'}
        )
    
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate recursive character splitter configuration."""
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if config.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        # Validate separators if provided
        separators = config.params.get('separators')
        if separators is not None:
            if not isinstance(separators, list):
                raise ValueError("separators must be a list of strings")
            if not all(isinstance(sep, str) for sep in separators):
                raise ValueError("all separators must be strings") 