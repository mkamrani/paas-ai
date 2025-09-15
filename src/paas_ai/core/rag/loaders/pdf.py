"""
PDF document loader strategy.
"""

import os
from typing import Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class PDFLoaderStrategy(LoaderStrategy):
    """Strategy for loading PDF documents with citation-friendly metadata."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a PDF document loader with enhanced metadata extraction."""
        params = config.params.copy()
        
        # Use PyPDFLoader which preserves page numbers
        loader = PyPDFLoader(
            file_path=url,
            **params
        )
        
        return loader
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate PDF loader configuration."""
        # Check if it's a local file
        if not url.startswith(('http://', 'https://')):
            if not os.path.exists(url):
                raise ValueError(f"PDF file not found: {url}")
            if not url.lower().endswith('.pdf'):
                raise ValueError(f"File does not have .pdf extension: {url}")
        elif not url.lower().endswith('.pdf'):
            raise ValueError(f"URL does not point to a PDF file: {url}") 