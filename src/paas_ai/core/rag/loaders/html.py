"""
HTML document loader strategy.
"""

import os
from typing import Dict, Any
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class HTMLLoaderStrategy(LoaderStrategy):
    """Strategy for loading HTML documents."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create an HTML document loader."""
        params = config.params.copy()
        return UnstructuredHTMLLoader(
            file_path=url,
            **params
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate HTML loader configuration."""
        # Check if it's a local file
        if not url.startswith(('http://', 'https://')):
            if not os.path.exists(url):
                raise ValueError(f"HTML file not found: {url}")
            if not url.lower().endswith(('.html', '.htm')):
                raise ValueError(f"File does not have .html/.htm extension: {url}") 