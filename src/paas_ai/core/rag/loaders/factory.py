"""
Factory for creating document loaders using strategy pattern.
"""

from typing import Dict, Any
from langchain_core.document_loaders import BaseLoader

from ..config import LoaderConfig, LoaderType
from .registry import LoaderRegistry


class DocumentLoaderFactory:
    """Factory for creating document loaders using strategy pattern."""
    
    @staticmethod
    def create_loader(config: LoaderConfig, url: str) -> BaseLoader:
        """Create a document loader based on configuration."""
        # Get strategy for this loader type
        strategy = LoaderRegistry.get_strategy(config.type)
        
        # Validate configuration
        strategy.validate_config(config, url)
        
        # Create and return loader
        return strategy.create_loader(config, url)
    
    @staticmethod
    def list_supported_types() -> list[LoaderType]:
        """List all supported loader types."""
        return LoaderRegistry.list_supported_types()
    
    @staticmethod 
    def register_strategy(loader_type: LoaderType, strategy_class) -> None:
        """Register a custom strategy for a loader type."""
        LoaderRegistry.register_strategy(loader_type, strategy_class) 