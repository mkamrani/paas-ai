"""
Markdown document loader strategy.
"""

import os
from typing import Dict, Any
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class MarkdownLoaderStrategy(LoaderStrategy):
    """Strategy for loading Markdown documents."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a Markdown document loader."""
        params = config.params.copy()
        return UnstructuredMarkdownLoader(
            file_path=url,
            **params
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate Markdown loader configuration."""
        # Check if it's a local file
        if not url.startswith(('http://', 'https://')):
            if not os.path.exists(url):
                raise ValueError(f"Markdown file not found: {url}")
            if not url.lower().endswith(('.md', '.markdown')):
                raise ValueError(f"File does not have .md/.markdown extension: {url}") 