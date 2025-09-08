"""
Factory for creating text splitters using strategy pattern.
"""

from typing import List
from langchain_core.documents import Document

from ..config import SplitterConfig, SplitterType
from .registry import SplitterRegistry


class TextSplitterFactory:
    """Factory for creating text splitters using strategy pattern."""
    
    @staticmethod
    def create_splitter(config: SplitterConfig):
        """Create a text splitter based on configuration."""
        # Get strategy for this splitter type
        strategy = SplitterRegistry.get_strategy(config.type)
        
        # Validate configuration
        strategy.validate_config(config)
        
        # Create and return splitter
        return strategy.create_splitter(config)
    
    @staticmethod
    def split_documents(config: SplitterConfig, documents: List[Document]) -> List[Document]:
        """Split documents using the configured splitter strategy."""
        strategy = SplitterRegistry.get_strategy(config.type)
        strategy.validate_config(config)
        return strategy.split_documents(config, documents)
    
    @staticmethod
    def list_supported_types() -> list[SplitterType]:
        """List all supported splitter types."""
        return SplitterRegistry.list_supported_types()
    
    @staticmethod 
    def register_strategy(splitter_type: SplitterType, strategy_class) -> None:
        """Register a custom strategy for a splitter type."""
        SplitterRegistry.register_strategy(splitter_type, strategy_class) 