"""
Web document loader strategy.
"""

from typing import Dict, Any
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class WebLoaderStrategy(LoaderStrategy):
    """Strategy for loading web documents with citation-friendly metadata."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a web document loader with enhanced metadata extraction."""
        params = config.params.copy()
        
        return WebBaseLoader(
            web_paths=[url],
            **params
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate web loader configuration."""
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Web loader requires HTTP/HTTPS URL, got: {url}") 