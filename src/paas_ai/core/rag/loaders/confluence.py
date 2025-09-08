"""
Confluence document loader strategy.
"""

from typing import Dict, Any
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class ConfluenceLoaderStrategy(LoaderStrategy):
    """Strategy for loading Confluence documents."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a Confluence document loader."""
        try:
            from langchain_community.document_loaders import ConfluenceLoader
        except ImportError:
            raise ImportError("Confluence loader requires additional dependencies. Install with: pip install atlassian-python-api")
        
        params = config.params.copy()
        return ConfluenceLoader(
            url=params.get('base_url', url),
            username=params.get('username'),
            api_key=params.get('api_key'),
            **{k: v for k, v in params.items() if k not in ['base_url', 'username', 'api_key']}
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate Confluence loader configuration."""
        params = config.params
        
        # Check required credentials
        if not params.get('username'):
            raise ValueError("Confluence loader requires 'username' in config params")
        if not params.get('api_key'):
            raise ValueError("Confluence loader requires 'api_key' in config params")
            
        # Validate URL format
        base_url = params.get('base_url', url)
        if not base_url.startswith(('http://', 'https://')):
            raise ValueError(f"Confluence base_url must be HTTP/HTTPS URL, got: {base_url}")
        if 'confluence' not in base_url.lower():
            raise ValueError(f"URL does not appear to be a Confluence instance: {base_url}") 