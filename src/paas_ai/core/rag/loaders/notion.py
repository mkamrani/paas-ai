"""
Notion document loader strategy.
"""

import os
from typing import Dict, Any
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class NotionLoaderStrategy(LoaderStrategy):
    """Strategy for loading Notion documents."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a Notion document loader."""
        try:
            from langchain_community.document_loaders import NotionDirectoryLoader
        except ImportError:
            raise ImportError("Notion loader requires additional dependencies. Install with: pip install notion-client")
        
        params = config.params.copy()
        return NotionDirectoryLoader(url, **params)
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate Notion loader configuration."""
        # For Notion, URL should be a local directory path containing exported Notion files
        if not os.path.exists(url):
            raise ValueError(f"Notion export directory not found: {url}")
        if not os.path.isdir(url):
            raise ValueError(f"Notion URL must be a directory path, got: {url}")
            
        # Check if directory contains any files
        if not os.listdir(url):
            raise ValueError(f"Notion directory is empty: {url}")
            
        # Check for typical Notion export files
        has_notion_files = any(
            f.endswith(('.md', '.html', '.csv')) 
            for f in os.listdir(url)
        )
        if not has_notion_files:
            raise ValueError(f"Directory does not contain typical Notion export files (.md, .html, .csv): {url}") 