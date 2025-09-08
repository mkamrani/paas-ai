"""
Character text splitter strategy.
"""

from langchain_text_splitters import CharacterTextSplitter

from .base import SplitterStrategy
from ..config import SplitterConfig


class CharacterSplitterStrategy(SplitterStrategy):
    """Strategy for character-based text splitting."""
    
    def create_splitter(self, config: SplitterConfig):
        """Create a character text splitter."""
        params = config.params.copy()
        return CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator=params.get('separator', '\n\n'),
            **{k: v for k, v in params.items() if k != 'separator'}
        )
    
    def validate_config(self, config: SplitterConfig) -> None:
        """Validate character splitter configuration."""
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if config.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size") 