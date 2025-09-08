"""
CSV document loader strategy.
"""

import os
import csv
from typing import Dict, Any
from langchain_community.document_loaders import CSVLoader
from langchain_core.document_loaders import BaseLoader

from .base import LoaderStrategy
from ..config import LoaderConfig


class CSVLoaderStrategy(LoaderStrategy):
    """Strategy for loading CSV documents."""
    
    def create_loader(self, config: LoaderConfig, url: str) -> BaseLoader:
        """Create a CSV document loader."""
        params = config.params.copy()
        return CSVLoader(
            file_path=url,
            **params
        )
    
    def validate_config(self, config: LoaderConfig, url: str) -> None:
        """Validate CSV loader configuration."""
        # Check if it's a local file
        if not url.startswith(('http://', 'https://')):
            if not os.path.exists(url):
                raise ValueError(f"CSV file not found: {url}")
            if not url.lower().endswith('.csv'):
                raise ValueError(f"File does not have .csv extension: {url}")
                
            # Validate CSV format if it's a local file
            try:
                with open(url, 'r') as f:
                    csv.Sniffer().sniff(f.read(1024))
            except csv.Error as e:
                raise ValueError(f"Invalid CSV file: {url} - {e}")
        elif not url.lower().endswith('.csv'):
            raise ValueError(f"URL does not point to a CSV file: {url}") 