"""
Base strategy interface for document loaders.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.document_loaders import BaseLoader

from ..config import LoaderConfig


class LoaderStrategy(ABC):
    """Base strategy interface for document loaders."""
    
    @abstractmethod
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a document loader based on configuration and URL."""
        pass
    
    @abstractmethod
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate configuration and URL for this loader strategy."""
        pass 