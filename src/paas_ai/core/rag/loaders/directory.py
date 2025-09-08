"""
Directory document loader strategy.
"""

import os
from typing import Dict, Any
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class DirectoryLoaderStrategy(LoaderStrategy):
    """Strategy for loading documents from directories."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a directory document loader."""
        params = config.params.copy()
        return DirectoryLoader(
            path=url,
            glob=params.get('glob', '**/*'),
            loader_cls=params.get('loader_cls'),
            **{k: v for k, v in params.items() if k not in ['glob', 'loader_cls']}
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate directory loader configuration."""
        if not os.path.exists(url):
            raise ValueError(f"Directory not found: {url}")
        if not os.path.isdir(url):
            raise ValueError(f"Path is not a directory: {url}")
            
        # Check if directory has any files matching the glob pattern
        import glob
        pattern = config.params.get('glob', '**/*')
        full_pattern = os.path.join(url, pattern)
        
        if not glob.glob(full_pattern, recursive=True):
            raise ValueError(f"No files found in directory {url} matching pattern {pattern}") 